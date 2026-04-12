"""
OptiSpark Plan Analyzer — Structured Spark execution plan parser and rule-based diagnostics.

Replaces brittle string-matching heuristics with a deterministic system that:
1. Parses Spark explain() output into typed PlanNode trees
2. Runs configurable detection rules over the tree
3. Produces structured PlanDiagnostic findings for the LLM context
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Plan Node Representation
# ═══════════════════════════════════════════════════════════════════════════════

# Recognized Catalyst physical plan node types.
# The parser maps raw plan text against these to classify nodes.
KNOWN_NODE_TYPES = {
    "Exchange",
    "BroadcastExchange",
    "BroadcastHashJoin",
    "SortMergeJoin",
    "ShuffledHashJoin",
    "CartesianProduct",
    "BroadcastNestedLoopJoin",
    "HashAggregate",
    "ObjectHashAggregate",
    "SortAggregate",
    "Sort",
    "Filter",
    "Project",
    "Scan",                     # Covers FileScan parquet/csv/json/orc
    "Repartition",
    "RepartitionByExpression",
    "Coalesce",
    "Expand",
    "Generate",                 # explode / posexplode / inline
    "Window",
    "Union",
    "LocalTableScan",
    "InMemoryTableScan",
    "SubqueryBroadcast",
    "AdaptiveSparkPlan",
    "CustomShuffleReader",
    "AQEShuffleRead",
    "CollectLimit",
    "TakeOrderedAndProject",
    "WholeStageCodegen",
}


@dataclass
class PlanNode:
    """A single node in a parsed Spark execution plan tree."""

    node_type: str
    raw_text: str
    depth: int = 0
    children: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    def walk(self):
        """Depth-first traversal yielding every node in the subtree."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find_all(self, node_type: str):
        """Return all nodes of a given type in this subtree."""
        return [n for n in self.walk() if n.node_type == node_type]

    def __repr__(self):
        child_count = len(self.children)
        return f"PlanNode({self.node_type}, children={child_count})"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


# ═══════════════════════════════════════════════════════════════════════════════
# Plan Parser
# ═══════════════════════════════════════════════════════════════════════════════

# Regex to detect the leading tree-drawing characters in Spark's explain output.
# Examples: "+- ", ":- ", ":  +- ", "   +- "
_INDENT_PATTERN = re.compile(r'^([+:|!\s-]*)(\S.*)')

# Regex to pull the node type token from the start of a plan line.
# Handles lines like "BroadcastHashJoin [user_id#10], ..." or "*(2) HashAggregate(...)"
_NODE_TYPE_PATTERN = re.compile(
    r'(?:\*\(\d+\)\s*)?'           # optional WholeStageCodegen marker like *(2)
    r'([A-Z][A-Za-z]*(?:\s[A-Za-z]+)?)'  # PascalCase node type (1-2 words)
)

# Regex for extracting join keys from join nodes.
_JOIN_KEYS_PATTERN = re.compile(r'\[([^\]]+)\]')

# Regex for FileScan variants.
_SCAN_PATTERN = re.compile(r'(?:File)?Scan\s+(\w+)', re.IGNORECASE)


def _classify_node(text: str) -> str:
    """Determine the canonical node type from a plan line's text."""
    stripped = text.strip()

    # Strip optional WholeStageCodegen prefix like "*(2) "
    cleaned = re.sub(r'^\*\(\d+\)\s*', '', stripped)

    # Priority 1: direct prefix match against known types (longest match wins).
    # This handles "BroadcastExchange HashedRelation..." correctly by matching
    # "BroadcastExchange" before the shorter "Exchange".
    best_match = None
    for known in KNOWN_NODE_TYPES:
        if cleaned.startswith(known):
            if best_match is None or len(known) > len(best_match):
                best_match = known
    if best_match:
        return best_match

    # Priority 2: Handle Scan variants (FileScan parquet, Scan csv, etc.)
    scan_match = _SCAN_PATTERN.search(cleaned)
    if scan_match:
        return "Scan"

    # Priority 3: Substring match (handles nodes deep in the line text)
    for known in sorted(KNOWN_NODE_TYPES, key=len, reverse=True):
        if known in cleaned:
            return known

    # Priority 4: Regex fallback — extract a PascalCase token from the line.
    regex_match = _NODE_TYPE_PATTERN.match(cleaned)
    if regex_match:
        return regex_match.group(1)

    return "Unknown"


def _parse_attributes(text: str, node_type: str) -> dict:
    """Extract structured attributes from a plan node's raw text."""
    attrs = {}

    # Join keys
    if "Join" in node_type:
        keys = _JOIN_KEYS_PATTERN.findall(text)
        if keys:
            attrs["join_keys"] = keys
        # Join type (Inner, LeftOuter, etc.)
        for jt in ("Inner", "LeftOuter", "RightOuter", "FullOuter", "LeftSemi", "LeftAnti", "Cross"):
            if jt in text:
                attrs["join_type"] = jt
                break

    # Repartition partition count
    if "Repartition" in node_type or node_type == "Exchange":
        num_match = re.search(r'(\d+)(?:\s*$|\s*,)', text)
        if num_match:
            attrs["num_partitions"] = int(num_match.group(1))

    # Exchange type (hashpartitioning, rangepartitioning, etc.)
    if node_type == "Exchange":
        if "hashpartitioning" in text.lower():
            attrs["exchange_type"] = "hash"
        elif "rangepartitioning" in text.lower():
            attrs["exchange_type"] = "range"
        elif "roundrobinpartitioning" in text.lower():
            attrs["exchange_type"] = "roundrobin"
        elif "singlepartition" in text.lower():
            attrs["exchange_type"] = "single"

    # Scan format
    if node_type == "Scan":
        scan_match = _SCAN_PATTERN.search(text)
        if scan_match:
            attrs["format"] = scan_match.group(1).lower()

    return attrs


def parse_plan(plan_text: str) -> list:
    """Parse a Spark explain() output into a forest of PlanNode trees.

    Uses indentation depth (measured by leading tree-drawing characters) to
    reconstruct the parent-child relationships from Spark's text format.

    Args:
        plan_text: Raw output from df.explain() or df._jdf.queryExecution().toString()

    Returns:
        A list of root PlanNode instances. Typically one root for simple plans,
        potentially multiple for union/subquery plans.
    """
    if not plan_text or not plan_text.strip():
        return []

    lines = plan_text.strip().splitlines()
    nodes = []

    for line in lines:
        if not line.strip():
            continue

        match = _INDENT_PATTERN.match(line)
        if not match:
            continue

        indent_chars = match.group(1)
        content = match.group(2)

        # Depth is determined by the length of leading tree-drawing characters.
        # Normalize: each 3 characters of indent = 1 depth level.
        depth = len(indent_chars) // 3

        node_type = _classify_node(content)
        attributes = _parse_attributes(content, node_type)

        node = PlanNode(
            node_type=node_type,
            raw_text=content.strip(),
            depth=depth,
            attributes=attributes,
        )
        nodes.append(node)

    # Reconstruct the tree from the flat depth-annotated list.
    return _build_tree(nodes)


def _build_tree(flat_nodes: list) -> list:
    """Reconstruct parent-child relationships from a depth-ordered node list."""
    if not flat_nodes:
        return []

    roots = []
    stack = []  # Stack of (depth, node) tuples

    for node in flat_nodes:
        # Pop stack until we find the parent (a node at depth - 1)
        while stack and stack[-1][0] >= node.depth:
            stack.pop()

        if stack:
            parent = stack[-1][1]
            parent.children.append(node)
        else:
            roots.append(node)

        stack.append((node.depth, node))

    return roots


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic Representation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanDiagnostic:
    """A single finding from analyzing an execution plan."""

    rule_id: str
    severity: str       # "INFO", "WARNING", "CRITICAL"
    message: str
    suggestion: str
    node: Optional[PlanNode] = None

    def to_dict(self) -> dict:
        """Serialize for JSON transport to the LLM context."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Detection Rules
# ═══════════════════════════════════════════════════════════════════════════════

def _rule_missing_broadcast(all_nodes: list, broadcast_threshold_bytes: int) -> list:
    """Detect SortMergeJoin nodes where broadcasting might be more efficient.

    A SortMergeJoin triggers a full shuffle of both sides. If one side's scan
    is small enough to fit within the broadcast threshold, a BroadcastHashJoin
    would eliminate the shuffle entirely.
    """
    diagnostics = []
    for node in all_nodes:
        if node.node_type == "SortMergeJoin":
            join_keys = node.attributes.get("join_keys", [])
            key_str = ", ".join(join_keys) if join_keys else "unknown keys"
            diagnostics.append(PlanDiagnostic(
                rule_id="MISSING_BROADCAST",
                severity="WARNING",
                message=(
                    f"SortMergeJoin detected on [{key_str}]. This triggers a full shuffle of both "
                    f"sides. If either side is smaller than the broadcast threshold "
                    f"({broadcast_threshold_bytes / (1024*1024):.0f} MB), a BroadcastHashJoin "
                    f"would eliminate the shuffle."
                ),
                suggestion="Use F.broadcast() on the smaller DataFrame in the join.",
                node=node,
            ))
    return diagnostics


def _rule_excessive_shuffle(all_nodes: list) -> list:
    """Detect consecutive Exchange nodes indicating redundant shuffles."""
    diagnostics = []
    for node in all_nodes:
        if node.node_type == "Exchange":
            for child in node.children:
                if child.node_type == "Exchange":
                    diagnostics.append(PlanDiagnostic(
                        rule_id="EXCESSIVE_SHUFFLE",
                        severity="WARNING",
                        message=(
                            "Consecutive Exchange (shuffle) nodes detected. The inner shuffle "
                            "is immediately re-shuffled by the outer Exchange, wasting network I/O."
                        ),
                        suggestion=(
                            "Remove the redundant .repartition() or coalesce() call, "
                            "or restructure the pipeline so both operations share a single shuffle."
                        ),
                        node=node,
                    ))
    return diagnostics


def _rule_redundant_repartition(all_nodes: list) -> list:
    """Detect HashAggregate whose child is a Repartition on the same keys.
    
    In Spark's plan tree, the aggregate is the parent consuming data from
    the repartition child. Since groupBy() triggers its own shuffle, the
    explicit repartition is a wasted shuffle stage.
    """
    diagnostics = []
    for node in all_nodes:
        if node.node_type == "HashAggregate":
            for child in node.children:
                if child.node_type in ("RepartitionByExpression", "Exchange"):
                    diagnostics.append(PlanDiagnostic(
                        rule_id="REDUNDANT_REPARTITION",
                        severity="INFO",
                        message=(
                            "A repartition/exchange feeds directly into a HashAggregate. "
                            "Since groupBy() triggers its own shuffle, the explicit repartition "
                            "is a wasted shuffle stage."
                        ),
                        suggestion="Remove the .repartition() call before the groupBy().",
                        node=child,
                    ))
    return diagnostics


def _rule_sort_merge_join_info(all_nodes: list) -> list:
    """Informational: flag all SortMergeJoin nodes for visibility."""
    diagnostics = []
    for node in all_nodes:
        if node.node_type == "SortMergeJoin":
            join_keys = node.attributes.get("join_keys", [])
            key_str = ", ".join(join_keys) if join_keys else "unknown keys"
            diagnostics.append(PlanDiagnostic(
                rule_id="SORT_MERGE_JOIN",
                severity="INFO",
                message=f"SortMergeJoin on [{key_str}] — requires sorting and shuffling both sides.",
                suggestion="Consider if either side is small enough to broadcast.",
                node=node,
            ))
    return diagnostics


def analyze_plan(roots: list, broadcast_threshold_bytes: int = 10 * 1024 * 1024) -> list:
    """Run all detection rules over a parsed plan tree.

    Args:
        roots: List of root PlanNode instances from parse_plan().
        broadcast_threshold_bytes: The autoBroadcastJoinThreshold from the SparkSession.
            Defaults to Spark's default of 10 MB.

    Returns:
        A list of PlanDiagnostic instances.  ``SORT_MERGE_JOIN`` informational
        findings are suppressed for nodes that already produced a
        ``MISSING_BROADCAST`` warning, but multiple findings with the same
        ``rule_id`` may still appear when multiple matching nodes exist.
    """
    # Flatten the forest into a single list for rule functions.
    all_nodes = []
    for root in roots:
        all_nodes.extend(root.walk())

    diagnostics = []
    diagnostics.extend(_rule_missing_broadcast(all_nodes, broadcast_threshold_bytes))
    diagnostics.extend(_rule_excessive_shuffle(all_nodes))
    diagnostics.extend(_rule_redundant_repartition(all_nodes))
    # Only add informational SMJ rule if MISSING_BROADCAST didn't already flag it
    broadcast_flagged = {d.node for d in diagnostics if d.rule_id == "MISSING_BROADCAST"}
    for d in _rule_sort_merge_join_info(all_nodes):
        if d.node not in broadcast_flagged:
            diagnostics.append(d)

    return diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# Debug Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_plan_tree(roots: list) -> None:
    """Pretty-print the parsed plan tree for debugging."""
    def _print_node(node: PlanNode, prefix: str = "", is_last: bool = True):
        connector = "└─ " if is_last else "├─ "
        attrs_str = ""
        if node.attributes:
            attrs_str = f"  {node.attributes}"
        print(f"{prefix}{connector}[{node.node_type}]{attrs_str}")

        child_prefix = prefix + ("   " if is_last else "│  ")
        for i, child in enumerate(node.children):
            _print_node(child, child_prefix, i == len(node.children) - 1)

    for i, root in enumerate(roots):
        _print_node(root, is_last=(i == len(roots) - 1))


def print_diagnostics(diagnostics: list) -> None:
    """Print diagnostics in a readable debug format."""
    if not diagnostics:
        print("  No diagnostics found.")
        return

    for i, d in enumerate(diagnostics, 1):
        severity_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🔴"}.get(d.severity, "?")
        print(f"  {severity_icon} [{d.rule_id}] ({d.severity})")
        print(f"     {d.message}")
        print(f"     → {d.suggestion}")
        if i < len(diagnostics):
            print()
