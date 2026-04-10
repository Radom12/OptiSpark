"""
OptiSpark Validator — Correctness validation for optimized DataFrames.

Compares original and optimized DataFrames across schema, row count,
data integrity, and aggregate parity. Produces a ValidationResult with
a confidence score that determines whether the optimization is accepted
or rejected.

Part of OptiSpark's self-correcting reasoning loop: when validation fails,
build_failure_prompt() constructs a structured prompt for the LLM to
diagnose and fix the semantic correctness issue.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    """Outcome of comparing an original DataFrame against an optimized one."""

    passed: bool
    checks: list = field(default_factory=list)
    sample_diffs: Optional[dict] = None
    confidence: str = "LOW"           # "HIGH", "MEDIUM", "LOW"
    confidence_score: float = 0.0     # 0.0 - 1.0

    def to_dict(self) -> dict:
        """Serialize for logging / LLM context injection."""
        return {
            "passed": self.passed,
            "checks": self.checks,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "sample_diffs": self.sample_diffs,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Core Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_optimization(
    original_df,
    optimized_df,
    sample_size: int = 1000,
    aggregate_tolerance: float = 0.0001,
    attempt_number: int = 1,
) -> ValidationResult:
    """Validate that an optimized DataFrame is semantically equivalent to the original.

    Runs four checks sequentially. Schema and row-count failures are critical
    and short-circuit further validation. Data integrity and aggregate parity
    provide additional confidence.

    Args:
        original_df: The original, unmodified PySpark DataFrame.
        optimized_df: The LLM-generated optimized DataFrame.
        sample_size: Number of rows to sample for comparison (default 1000).
        aggregate_tolerance: Relative tolerance for numeric aggregate comparison (default 0.01%).
        attempt_number: Which retry attempt this is (1-based). Affects confidence scoring.

    Returns:
        A ValidationResult with pass/fail status, individual check results,
        confidence score, and any detected differences.
    """
    checks = []
    sample_diffs = None

    # ── Check 1: Schema Match ─────────────────────────────────────────────
    schema_result = _check_schema(original_df, optimized_df)
    checks.append(schema_result)
    if not schema_result["passed"]:
        confidence, score = _compute_confidence(checks, attempt_number)
        return ValidationResult(
            passed=False, checks=checks, confidence=confidence,
            confidence_score=score,
        )

    # ── Check 2: Row Count (on sample) ────────────────────────────────────
    row_result = _check_row_count(original_df, optimized_df, sample_size)
    checks.append(row_result)
    if not row_result["passed"]:
        confidence, score = _compute_confidence(checks, attempt_number)
        return ValidationResult(
            passed=False, checks=checks, confidence=confidence,
            confidence_score=score,
        )

    # ── Check 3: Data Integrity (subtract) ────────────────────────────────
    integrity_result, sample_diffs = _check_data_integrity(
        original_df, optimized_df, sample_size
    )
    checks.append(integrity_result)

    # ── Check 4: Aggregate Parity ─────────────────────────────────────────
    agg_result = _check_aggregate_parity(
        original_df, optimized_df, sample_size, aggregate_tolerance
    )
    checks.append(agg_result)

    # ── Compute Final Result ──────────────────────────────────────────────
    all_passed = all(c["passed"] for c in checks)
    confidence, score = _compute_confidence(checks, attempt_number)

    return ValidationResult(
        passed=all_passed,
        checks=checks,
        sample_diffs=sample_diffs,
        confidence=confidence,
        confidence_score=score,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Checks
# ═══════════════════════════════════════════════════════════════════════════════

def _check_schema(original_df, optimized_df) -> dict:
    """Verify column names and data types match exactly."""
    orig_fields = [(f.name, str(f.dataType)) for f in original_df.schema.fields]
    opt_fields = [(f.name, str(f.dataType)) for f in optimized_df.schema.fields]

    if orig_fields == opt_fields:
        return {
            "name": "schema_match",
            "passed": True,
            "detail": f"Schema matches ({len(orig_fields)} columns).",
        }

    # Build a detailed mismatch report
    orig_names = [f[0] for f in orig_fields]
    opt_names = [f[0] for f in opt_fields]

    missing = set(orig_names) - set(opt_names)
    extra = set(opt_names) - set(orig_names)
    type_mismatches = []

    for name, dtype in orig_fields:
        for oname, otype in opt_fields:
            if name == oname and dtype != otype:
                type_mismatches.append(f"{name}: {dtype} → {otype}")

    parts = []
    if missing:
        parts.append(f"missing columns: {sorted(missing)}")
    if extra:
        parts.append(f"extra columns: {sorted(extra)}")
    if type_mismatches:
        parts.append(f"type mismatches: {type_mismatches}")

    return {
        "name": "schema_match",
        "passed": False,
        "detail": "Schema mismatch — " + "; ".join(parts),
    }


def _check_row_count(original_df, optimized_df, sample_size: int) -> dict:
    """Compare row counts on a limited sample."""
    try:
        orig_count = original_df.limit(sample_size).count()
        opt_count = optimized_df.limit(sample_size).count()

        if orig_count == opt_count:
            return {
                "name": "row_count",
                "passed": True,
                "detail": f"Row counts match ({orig_count} == {opt_count}).",
            }
        return {
            "name": "row_count",
            "passed": False,
            "detail": f"Row count mismatch: original={orig_count}, optimized={opt_count}.",
        }
    except Exception as e:
        return {
            "name": "row_count",
            "passed": False,
            "detail": f"Row count check failed with error: {str(e)}",
        }


def _check_data_integrity(original_df, optimized_df, sample_size: int) -> tuple:
    """Use subtract() to detect row-level differences on a sample.

    Returns (check_dict, sample_diffs_dict_or_None).
    """
    try:
        orig_sample = original_df.limit(sample_size)
        opt_sample = optimized_df.limit(sample_size)

        # Rows in original not in optimized
        only_in_orig = orig_sample.subtract(opt_sample)
        orig_diff_count = only_in_orig.count()

        # Rows in optimized not in original
        only_in_opt = opt_sample.subtract(orig_sample)
        opt_diff_count = only_in_opt.count()

        if orig_diff_count == 0 and opt_diff_count == 0:
            return (
                {
                    "name": "data_integrity",
                    "passed": True,
                    "detail": "No row-level differences detected in sample.",
                },
                None,
            )

        # Collect a small sample of differences for the failure prompt
        diff_sample = {}
        if orig_diff_count > 0:
            diff_sample["only_in_original"] = [
                row.asDict() for row in only_in_orig.limit(5).collect()
            ]
        if opt_diff_count > 0:
            diff_sample["only_in_optimized"] = [
                row.asDict() for row in only_in_opt.limit(5).collect()
            ]

        return (
            {
                "name": "data_integrity",
                "passed": False,
                "detail": (
                    f"Data mismatch: {orig_diff_count} rows only in original, "
                    f"{opt_diff_count} rows only in optimized."
                ),
            },
            diff_sample,
        )
    except Exception as e:
        return (
            {
                "name": "data_integrity",
                "passed": False,
                "detail": f"Data integrity check failed with error: {str(e)}",
            },
            None,
        )


def _check_aggregate_parity(
    original_df, optimized_df, sample_size: int, tolerance: float
) -> dict:
    """Compare basic aggregates (sum) for all numeric columns within tolerance."""
    try:
        from pyspark.sql import types as T

        numeric_types = (T.IntegerType, T.LongType, T.FloatType, T.DoubleType, T.DecimalType)
        numeric_cols = [
            f.name for f in original_df.schema.fields
            if isinstance(f.dataType, numeric_types)
        ]

        if not numeric_cols:
            return {
                "name": "aggregate_parity",
                "passed": True,
                "detail": "No numeric columns to compare.",
            }

        import pyspark.sql.functions as F

        orig_sample = original_df.limit(sample_size)
        opt_sample = optimized_df.limit(sample_size)

        # Build sum expressions for all numeric columns
        sum_exprs = [F.sum(F.col(c)).alias(c) for c in numeric_cols]

        orig_aggs = orig_sample.agg(*sum_exprs).collect()[0]
        opt_aggs = opt_sample.agg(*sum_exprs).collect()[0]

        mismatches = []
        for col_name in numeric_cols:
            orig_val = orig_aggs[col_name]
            opt_val = opt_aggs[col_name]

            # Both None — match
            if orig_val is None and opt_val is None:
                continue
            # One None — mismatch
            if orig_val is None or opt_val is None:
                mismatches.append(f"{col_name}: {orig_val} vs {opt_val}")
                continue
            # Relative difference check
            orig_f = float(orig_val)
            opt_f = float(opt_val)
            if orig_f == 0.0:
                if opt_f != 0.0:
                    mismatches.append(f"{col_name}: {orig_f} vs {opt_f}")
            else:
                rel_diff = abs(orig_f - opt_f) / abs(orig_f)
                if rel_diff > tolerance:
                    mismatches.append(
                        f"{col_name}: {orig_f} vs {opt_f} (diff={rel_diff:.6%})"
                    )

        if not mismatches:
            return {
                "name": "aggregate_parity",
                "passed": True,
                "detail": f"All {len(numeric_cols)} numeric columns match within {tolerance:.4%} tolerance.",
            }
        return {
            "name": "aggregate_parity",
            "passed": False,
            "detail": "Aggregate mismatches: " + "; ".join(mismatches),
        }
    except Exception as e:
        return {
            "name": "aggregate_parity",
            "passed": False,
            "detail": f"Aggregate parity check failed with error: {str(e)}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_confidence(checks: list, attempt_number: int) -> tuple:
    """Compute a confidence level and score based on check results and retry count.

    Returns:
        A tuple of (confidence_level: str, confidence_score: float).
    """
    all_passed = all(c["passed"] for c in checks)

    if not all_passed:
        return ("LOW", 0.0)

    if attempt_number == 1:
        return ("HIGH", 1.0)
    elif attempt_number == 2:
        return ("MEDIUM", 0.7)
    else:
        return ("MEDIUM", 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Failure-Aware Prompt Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_failure_prompt(
    validation_result: ValidationResult,
    original_code: str,
    plan_diagnostics: Optional[list] = None,
) -> str:
    """Construct a structured prompt for the LLM when validation fails.

    Includes the specific checks that failed, sample row differences,
    the code that was attempted, and instructions to fix the issue.

    Args:
        validation_result: The failed ValidationResult.
        original_code: The code block that produced the incorrect optimization.
        plan_diagnostics: Structured plan diagnostics from the plan analyzer (if available).

    Returns:
        A formatted string prompt for the LLM.
    """
    sections = []

    # Section 1: What failed
    sections.append("## Validation Failure Report")
    sections.append("")
    for check in validation_result.checks:
        status = "✔ PASSED" if check["passed"] else "✖ FAILED"
        sections.append(f"- **{check['name']}**: {status} — {check['detail']}")

    # Section 2: Sample differences
    if validation_result.sample_diffs:
        sections.append("")
        sections.append("## Sample Differences")
        diffs = validation_result.sample_diffs
        if "only_in_original" in diffs:
            sections.append(f"Rows only in original ({len(diffs['only_in_original'])} shown):")
            for row in diffs["only_in_original"][:3]:
                sections.append(f"  {row}")
        if "only_in_optimized" in diffs:
            sections.append(f"Rows only in optimized ({len(diffs['only_in_optimized'])} shown):")
            for row in diffs["only_in_optimized"][:3]:
                sections.append(f"  {row}")

    # Section 3: The code that was attempted
    sections.append("")
    sections.append("## Code That Failed Validation")
    sections.append(f"```python\n{original_code.strip()}\n```")

    # Section 4: Plan diagnostics context
    if plan_diagnostics:
        sections.append("")
        sections.append("## Plan Diagnostics That Motivated This Optimization")
        for diag in plan_diagnostics[:3]:
            sections.append(f"- [{diag.get('rule_id', '?')}] {diag.get('message', '')}")

    # Section 5: Instructions
    sections.append("")
    sections.append("## Instructions")
    sections.append(
        "The optimization above executed successfully but produced INCORRECT results. "
        "The optimized DataFrame does not match the original DataFrame semantically. "
        "You must diagnose the root cause of the data mismatch and generate a corrected "
        "version that preserves the exact same output schema and data. "
        "Assign the corrected result to `optimized_df`."
    )

    return "\n".join(sections)
