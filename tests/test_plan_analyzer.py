"""Tests for the structured execution plan analyzer."""

import pytest
from optispark.plan_analyzer import (
    PlanNode,
    PlanDiagnostic,
    parse_plan,
    analyze_plan,
    print_plan_tree,
    print_diagnostics,
    _classify_node,
    _parse_attributes,
    _build_tree,
)


# ─── Realistic Spark Plan Fixtures ───────────────────────────────────────────

SORT_MERGE_JOIN_PLAN = """\
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[user_id#10, region#20], functions=[sum(amount#15)])
   +- Exchange hashpartitioning(user_id#10, region#20, 200)
      +- HashAggregate(keys=[user_id#10, region#20], functions=[partial_sum(amount#15)])
         +- SortMergeJoin [user_id#10], [user_id#30], Inner
            :- Sort [user_id#10 ASC NULLS FIRST]
            :  +- Exchange hashpartitioning(user_id#10, 200)
            :     +- FileScan parquet [user_id#10, amount#15, region#20]
            +- Sort [user_id#30 ASC NULLS FIRST]
               +- Exchange hashpartitioning(user_id#30, 200)
                  +- FileScan parquet [user_id#30, name#35]
"""

BROADCAST_JOIN_PLAN = """\
== Physical Plan ==
*(2) HashAggregate(keys=[user_id#10], functions=[sum(amount#15)])
+- Exchange hashpartitioning(user_id#10, 200)
   +- *(1) HashAggregate(keys=[user_id#10], functions=[partial_sum(amount#15)])
      +- *(1) BroadcastHashJoin [user_id#10], [user_id#30], Inner, BuildRight
         :- *(1) FileScan parquet [user_id#10, amount#15]
         +- BroadcastExchange HashedRelationBroadcastMode
            +- *(0) FileScan parquet [user_id#30, name#35]
"""

CONSECUTIVE_EXCHANGE_PLAN = """\
== Physical Plan ==
+- Exchange hashpartitioning(user_id#10, 100)
   +- Exchange roundrobinpartitioning(200)
      +- FileScan parquet [user_id#10, amount#15]
"""

REPARTITION_BEFORE_AGGREGATE_PLAN = """\
== Physical Plan ==
+- HashAggregate(keys=[region#20], functions=[sum(amount#15)])
   +- Exchange hashpartitioning(region#20, 100)
      +- RepartitionByExpression [region#20], 100
         +- FileScan parquet [region#20, amount#15]
"""

SIMPLE_SCAN_PLAN = """\
== Physical Plan ==
+- FileScan parquet [user_id#10, amount#15]
"""

EMPTY_PLAN = ""
WHITESPACE_PLAN = "   \n\n  \n"


# ─── PlanNode Unit Tests ─────────────────────────────────────────────────────

class TestPlanNode:

    def test_walk_single_node(self):
        node = PlanNode(node_type="Scan", raw_text="FileScan parquet")
        walked = list(node.walk())
        assert len(walked) == 1
        assert walked[0] is node

    def test_walk_tree(self):
        child1 = PlanNode(node_type="Scan", raw_text="scan1")
        child2 = PlanNode(node_type="Scan", raw_text="scan2")
        parent = PlanNode(node_type="SortMergeJoin", raw_text="join", children=[child1, child2])
        walked = list(parent.walk())
        assert len(walked) == 3
        assert walked[0] is parent

    def test_find_all(self):
        scan1 = PlanNode(node_type="Scan", raw_text="scan1")
        scan2 = PlanNode(node_type="Scan", raw_text="scan2")
        join = PlanNode(node_type="SortMergeJoin", raw_text="join", children=[scan1, scan2])
        root = PlanNode(node_type="HashAggregate", raw_text="agg", children=[join])

        scans = root.find_all("Scan")
        assert len(scans) == 2
        assert root.find_all("Exchange") == []

    def test_repr(self):
        node = PlanNode(node_type="Filter", raw_text="f", children=[PlanNode(node_type="Scan", raw_text="s")])
        assert "Filter" in repr(node)
        assert "children=1" in repr(node)


# ─── Node Classifier Tests ──────────────────────────────────────────────────

class TestClassifyNode:

    def test_sort_merge_join(self):
        assert _classify_node("SortMergeJoin [user_id#10], [user_id#30], Inner") == "SortMergeJoin"

    def test_broadcast_hash_join(self):
        assert _classify_node("BroadcastHashJoin [id#1], [id#2], Inner, BuildRight") == "BroadcastHashJoin"

    def test_exchange(self):
        assert _classify_node("Exchange hashpartitioning(user_id#10, 200)") == "Exchange"

    def test_hash_aggregate(self):
        assert _classify_node("HashAggregate(keys=[user_id#10], functions=[sum(amount)])") == "HashAggregate"

    def test_file_scan_parquet(self):
        assert _classify_node("FileScan parquet [user_id#10, amount#15]") == "Scan"

    def test_file_scan_csv(self):
        assert _classify_node("Scan csv [col1#0, col2#1]") == "Scan"

    def test_wholestage_codegen(self):
        assert _classify_node("*(2) HashAggregate(keys=[id#10])") == "HashAggregate"

    def test_unknown_node(self):
        # A line that starts with a lowercase word cannot be classified by any
        # heuristic (prefix, substring, or PascalCase regex) — should be "Unknown".
        assert _classify_node("some_unrecognized_operator [args]") == "Unknown"

    def test_filter(self):
        assert _classify_node("Filter (id#10 > 5)") == "Filter"

    def test_sort(self):
        assert _classify_node("Sort [id#10 ASC NULLS FIRST]") == "Sort"

    def test_repartition_by_expression(self):
        assert _classify_node("RepartitionByExpression [region#20], 100") == "RepartitionByExpression"

    def test_broadcast_exchange(self):
        assert _classify_node("BroadcastExchange HashedRelationBroadcastMode") == "BroadcastExchange"


# ─── Attribute Parser Tests ─────────────────────────────────────────────────

class TestParseAttributes:

    def test_join_keys(self):
        attrs = _parse_attributes("SortMergeJoin [user_id#10], [user_id#30], Inner", "SortMergeJoin")
        assert "join_keys" in attrs
        assert "user_id#10" in attrs["join_keys"]

    def test_join_type(self):
        attrs = _parse_attributes("BroadcastHashJoin [id#1], [id#2], LeftOuter", "BroadcastHashJoin")
        assert attrs["join_type"] == "LeftOuter"

    def test_exchange_type_hash(self):
        attrs = _parse_attributes("Exchange hashpartitioning(user_id#10, 200)", "Exchange")
        assert attrs["exchange_type"] == "hash"

    def test_exchange_type_range(self):
        attrs = _parse_attributes("Exchange rangepartitioning(id#5, 100)", "Exchange")
        assert attrs["exchange_type"] == "range"

    def test_scan_format(self):
        attrs = _parse_attributes("FileScan parquet [col#1]", "Scan")
        assert attrs["format"] == "parquet"

    def test_no_attributes_for_filter(self):
        attrs = _parse_attributes("Filter (x > 5)", "Filter")
        assert attrs == {}


# ─── Plan Parser Tests ──────────────────────────────────────────────────────

class TestParsePlan:

    def test_empty_plan(self):
        assert parse_plan("") == []
        assert parse_plan(None) == []
        assert parse_plan("   \n\n  ") == []

    def test_simple_scan(self):
        nodes = parse_plan(SIMPLE_SCAN_PLAN)
        assert len(nodes) >= 1
        # Find at least one Scan node
        all_nodes = []
        for root in nodes:
            all_nodes.extend(root.walk())
        scan_nodes = [n for n in all_nodes if n.node_type == "Scan"]
        assert len(scan_nodes) >= 1

    def test_sort_merge_join_plan(self):
        roots = parse_plan(SORT_MERGE_JOIN_PLAN)
        assert len(roots) >= 1

        # Flatten all nodes
        all_nodes = []
        for root in roots:
            all_nodes.extend(root.walk())

        # Should have: SortMergeJoin, Exchange(s), Sort(s), Scan(s), HashAggregate(s)
        types = {n.node_type for n in all_nodes}
        assert "SortMergeJoin" in types
        assert "Exchange" in types
        assert "Scan" in types
        assert "HashAggregate" in types

    def test_broadcast_join_plan(self):
        roots = parse_plan(BROADCAST_JOIN_PLAN)
        all_nodes = []
        for root in roots:
            all_nodes.extend(root.walk())

        types = {n.node_type for n in all_nodes}
        assert "BroadcastHashJoin" in types
        assert "BroadcastExchange" in types

    def test_nodes_have_depth(self):
        roots = parse_plan(SORT_MERGE_JOIN_PLAN)
        # Root should be at depth 0, children deeper
        all_nodes = []
        for root in roots:
            all_nodes.extend(root.walk())
        depths = {n.depth for n in all_nodes}
        assert 0 in depths
        assert len(depths) > 1  # Multiple depths in a nested plan


# ─── Build Tree Tests ────────────────────────────────────────────────────────

class TestBuildTree:

    def test_empty(self):
        assert _build_tree([]) == []

    def test_single_node(self):
        node = PlanNode(node_type="Scan", raw_text="scan", depth=0)
        roots = _build_tree([node])
        assert len(roots) == 1
        assert roots[0] is node

    def test_parent_child_relationship(self):
        parent = PlanNode(node_type="Exchange", raw_text="exchange", depth=0)
        child = PlanNode(node_type="Scan", raw_text="scan", depth=1)
        roots = _build_tree([parent, child])
        assert len(roots) == 1
        assert len(roots[0].children) == 1
        assert roots[0].children[0] is child

    def test_two_siblings(self):
        parent = PlanNode(node_type="SortMergeJoin", raw_text="join", depth=0)
        child1 = PlanNode(node_type="Scan", raw_text="scan1", depth=1)
        child2 = PlanNode(node_type="Scan", raw_text="scan2", depth=1)
        roots = _build_tree([parent, child1, child2])
        assert len(roots) == 1
        assert len(roots[0].children) == 2


# ─── Rule Detection Tests ───────────────────────────────────────────────────

class TestAnalyzePlan:

    def test_missing_broadcast_detected(self):
        roots = parse_plan(SORT_MERGE_JOIN_PLAN)
        diagnostics = analyze_plan(roots, broadcast_threshold_bytes=10 * 1024 * 1024)
        rule_ids = {d.rule_id for d in diagnostics}
        assert "MISSING_BROADCAST" in rule_ids

    def test_missing_broadcast_uses_threshold(self):
        roots = parse_plan(SORT_MERGE_JOIN_PLAN)
        diagnostics = analyze_plan(roots, broadcast_threshold_bytes=50 * 1024 * 1024)
        # Should still flag, but message should reference 50 MB
        broadcast_diag = [d for d in diagnostics if d.rule_id == "MISSING_BROADCAST"]
        assert len(broadcast_diag) > 0
        assert "50 MB" in broadcast_diag[0].message

    def test_no_missing_broadcast_when_already_broadcast(self):
        roots = parse_plan(BROADCAST_JOIN_PLAN)
        diagnostics = analyze_plan(roots)
        rule_ids = {d.rule_id for d in diagnostics}
        assert "MISSING_BROADCAST" not in rule_ids

    def test_excessive_shuffle_detected(self):
        roots = parse_plan(CONSECUTIVE_EXCHANGE_PLAN)
        diagnostics = analyze_plan(roots)
        rule_ids = {d.rule_id for d in diagnostics}
        assert "EXCESSIVE_SHUFFLE" in rule_ids

    def test_no_excessive_shuffle_on_normal_plan(self):
        roots = parse_plan(BROADCAST_JOIN_PLAN)
        diagnostics = analyze_plan(roots)
        rule_ids = {d.rule_id for d in diagnostics}
        assert "EXCESSIVE_SHUFFLE" not in rule_ids

    def test_redundant_repartition_detected(self):
        roots = parse_plan(REPARTITION_BEFORE_AGGREGATE_PLAN)
        diagnostics = analyze_plan(roots)
        rule_ids = {d.rule_id for d in diagnostics}
        assert "REDUNDANT_REPARTITION" in rule_ids

    def test_empty_plan_no_diagnostics(self):
        roots = parse_plan("")
        diagnostics = analyze_plan(roots)
        assert diagnostics == []

    def test_sort_merge_join_not_duplicated_with_broadcast(self):
        """SORT_MERGE_JOIN info rule should not fire when MISSING_BROADCAST already flagged the same node."""
        roots = parse_plan(SORT_MERGE_JOIN_PLAN)
        diagnostics = analyze_plan(roots)
        smj_diags = [d for d in diagnostics if d.rule_id == "SORT_MERGE_JOIN"]
        # Should be 0 — deduplicated because MISSING_BROADCAST already covers it
        assert len(smj_diags) == 0


# ─── Diagnostic Serialization Tests ─────────────────────────────────────────

class TestPlanDiagnostic:

    def test_to_dict(self):
        d = PlanDiagnostic(
            rule_id="TEST_RULE",
            severity="WARNING",
            message="Something went wrong.",
            suggestion="Fix it.",
        )
        result = d.to_dict()
        assert result["rule_id"] == "TEST_RULE"
        assert result["severity"] == "WARNING"
        assert result["message"] == "Something went wrong."
        assert result["suggestion"] == "Fix it."

    def test_to_dict_no_node(self):
        d = PlanDiagnostic(
            rule_id="R1", severity="INFO", message="m", suggestion="s"
        )
        result = d.to_dict()
        # node is intentionally excluded from serialization
        assert "node" not in result


# ─── Debug Output Tests ─────────────────────────────────────────────────────

class TestDebugOutput:

    def test_print_plan_tree(self, capsys):
        roots = parse_plan(SORT_MERGE_JOIN_PLAN)
        print_plan_tree(roots)
        captured = capsys.readouterr()
        assert "SortMergeJoin" in captured.out
        assert "Scan" in captured.out

    def test_print_diagnostics(self, capsys):
        roots = parse_plan(SORT_MERGE_JOIN_PLAN)
        diagnostics = analyze_plan(roots)
        print_diagnostics(diagnostics)
        captured = capsys.readouterr()
        assert "MISSING_BROADCAST" in captured.out

    def test_print_diagnostics_empty(self, capsys):
        print_diagnostics([])
        captured = capsys.readouterr()
        assert "No diagnostics found" in captured.out
