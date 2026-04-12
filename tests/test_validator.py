"""Tests for the correctness validator module.

Uses MagicMock DataFrames for tests that trigger Spark actions (count, collect,
subtract) to avoid Py4J worker stability issues in CI. Schema tests use
the real Spark session since they only read metadata.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock
from pyspark.sql import types as T

from optispark.validator import (
    ValidationResult,
    validate_optimization,
    build_failure_prompt,
    _check_schema,
    _check_row_count,
    _check_data_integrity,
    _check_aggregate_parity,
    _compute_confidence,
)


# ─── Helper: Build a mock DataFrame with a given schema ──────────────────────

def _make_mock_df(fields, row_count=3, rows=None):
    """Create a MagicMock DataFrame with a real StructType schema.

    Args:
        fields: List of (name, pyspark_type) tuples.
        row_count: Number of rows limit().count() should return.
        rows: Optional list of dicts for collect() results.
    """
    schema = T.StructType([T.StructField(n, t, True) for n, t in fields])
    df = MagicMock()
    type(df).schema = PropertyMock(return_value=schema)

    # limit() returns a new mock that supports .count() and .collect()
    limited = MagicMock()
    limited.count.return_value = row_count
    limited.collect.return_value = [MagicMock(**{"asDict.return_value": r}) for r in (rows or [])]

    # exceptAll() returns a mock with count() and limit().collect()
    empty_exceptall = MagicMock()
    empty_exceptall.count.return_value = 0
    empty_exceptall.limit.return_value.collect.return_value = []
    limited.exceptAll.return_value = empty_exceptall

    df.limit.return_value = limited

    # agg() for aggregate parity — return a row with sum values
    agg_row = MagicMock()
    if rows:
        # Compute sums from the given rows
        agg_row.__getitem__ = MagicMock(side_effect=lambda key: sum(r.get(key, 0) for r in rows if r.get(key) is not None))
    limited.agg.return_value.collect.return_value = [agg_row]

    return df


# ─── ValidationResult Unit Tests ─────────────────────────────────────────────

class TestValidationResult:

    def test_to_dict(self):
        result = ValidationResult(
            passed=True,
            checks=[{"name": "schema_match", "passed": True, "detail": "ok"}],
            confidence="HIGH",
            confidence_score=1.0,
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["confidence"] == "HIGH"
        assert d["confidence_score"] == 1.0
        assert len(d["checks"]) == 1

    def test_defaults(self):
        result = ValidationResult(passed=False)
        assert result.confidence == "LOW"
        assert result.confidence_score == 0.0
        assert result.sample_diffs is None
        assert result.checks == []


# ─── Confidence Scoring Tests ────────────────────────────────────────────────

class TestConfidenceScoring:

    def test_all_passed_first_attempt(self):
        checks = [{"passed": True}, {"passed": True}]
        level, score = _compute_confidence(checks, attempt_number=1)
        assert level == "HIGH"
        assert score == 1.0

    def test_all_passed_second_attempt(self):
        checks = [{"passed": True}, {"passed": True}]
        level, score = _compute_confidence(checks, attempt_number=2)
        assert level == "MEDIUM"
        assert score == 0.7

    def test_all_passed_third_attempt(self):
        checks = [{"passed": True}]
        level, score = _compute_confidence(checks, attempt_number=3)
        assert level == "MEDIUM"
        assert score == 0.5

    def test_any_failed(self):
        checks = [{"passed": True}, {"passed": False}]
        level, score = _compute_confidence(checks, attempt_number=1)
        assert level == "LOW"
        assert score == 0.0


# ─── Schema Check Tests (with real Spark) ────────────────────────────────────

class TestSchemaCheck:

    def test_matching_schema(self, spark):
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
        df2 = spark.createDataFrame([(3, "c")], ["id", "name"])
        result = _check_schema(df, df2)
        assert result["passed"] is True

    def test_missing_column(self, spark):
        df1 = spark.createDataFrame([(1, "a")], ["id", "name"])
        df2 = spark.createDataFrame([(1,)], ["id"])
        result = _check_schema(df1, df2)
        assert result["passed"] is False
        assert "missing" in result["detail"].lower()

    def test_extra_column(self, spark):
        df1 = spark.createDataFrame([(1,)], ["id"])
        df2 = spark.createDataFrame([(1, "a")], ["id", "name"])
        result = _check_schema(df1, df2)
        assert result["passed"] is False
        assert "extra" in result["detail"].lower()

    def test_type_mismatch(self, spark):
        df1 = spark.createDataFrame([(1, "a")], ["id", "name"])
        df2 = spark.createDataFrame([("1", "a")], ["id", "name"])
        result = _check_schema(df1, df2)
        assert result["passed"] is False
        assert "type" in result["detail"].lower()

    def test_matching_schema_with_mock(self):
        """Schema check also works with mocked DataFrames."""
        df1 = _make_mock_df([("id", T.IntegerType()), ("name", T.StringType())])
        df2 = _make_mock_df([("id", T.IntegerType()), ("name", T.StringType())])
        result = _check_schema(df1, df2)
        assert result["passed"] is True

    def test_schema_mismatch_with_mock(self):
        df1 = _make_mock_df([("id", T.IntegerType()), ("name", T.StringType())])
        df2 = _make_mock_df([("id", T.IntegerType())])
        result = _check_schema(df1, df2)
        assert result["passed"] is False


# ─── Row Count Check Tests (mocked) ─────────────────────────────────────────

class TestRowCountCheck:

    def test_matching_count(self):
        df1 = _make_mock_df([("id", T.IntegerType())], row_count=100)
        df2 = _make_mock_df([("id", T.IntegerType())], row_count=100)
        result = _check_row_count(df1, df2, sample_size=1000)
        assert result["passed"] is True
        assert "100" in result["detail"]

    def test_both_at_sample_limit_passes_with_caveat(self):
        """When both counts equal sample_size the message reflects uncertainty."""
        df1 = _make_mock_df([("id", T.IntegerType())], row_count=500)
        df2 = _make_mock_df([("id", T.IntegerType())], row_count=500)
        result = _check_row_count(df1, df2, sample_size=500)
        assert result["passed"] is True
        assert "at least" in result["detail"]
        assert "500" in result["detail"]

    def test_mismatched_count(self):
        df1 = _make_mock_df([("id", T.IntegerType())], row_count=100)
        df2 = _make_mock_df([("id", T.IntegerType())], row_count=50)
        result = _check_row_count(df1, df2, sample_size=1000)
        assert result["passed"] is False
        assert "100" in result["detail"]
        assert "50" in result["detail"]

    def test_count_error_handled(self):
        df1 = MagicMock()
        df1.limit.return_value.count.side_effect = RuntimeError("Spark crashed")
        df2 = MagicMock()
        result = _check_row_count(df1, df2, sample_size=1000)
        assert result["passed"] is False
        assert "error" in result["detail"].lower()


# ─── Data Integrity Check Tests (mocked) ─────────────────────────────────────

class TestDataIntegrityCheck:

    def test_identical_data(self):
        df = _make_mock_df([("id", T.IntegerType())], row_count=3)
        result, diffs = _check_data_integrity(df, df, sample_size=1000)
        assert result["passed"] is True
        assert diffs is None

    def test_different_data(self):
        df1 = _make_mock_df([("id", T.IntegerType())], row_count=3,
                            rows=[{"id": 1}, {"id": 2}])
        df2 = _make_mock_df([("id", T.IntegerType())], row_count=3,
                            rows=[{"id": 1}, {"id": 99}])

        # Set up subtract to return non-empty results
        limited1 = df1.limit.return_value
        limited2 = df2.limit.return_value

        diff_result = MagicMock()
        diff_result.count.return_value = 1
        row_mock = MagicMock()
        row_mock.asDict.return_value = {"id": 2}
        diff_result.limit.return_value.collect.return_value = [row_mock]
        limited1.exceptAll.return_value = diff_result

        diff_result2 = MagicMock()
        diff_result2.count.return_value = 1
        row_mock2 = MagicMock()
        row_mock2.asDict.return_value = {"id": 99}
        diff_result2.limit.return_value.collect.return_value = [row_mock2]
        limited2.exceptAll.return_value = diff_result2

        result, diffs = _check_data_integrity(df1, df2, sample_size=1000)
        assert result["passed"] is False
        assert diffs is not None
        assert "only_in_original" in diffs
        assert "only_in_optimized" in diffs

    def test_integrity_error_handled(self):
        df1 = MagicMock()
        df1.limit.side_effect = RuntimeError("boom")
        df2 = MagicMock()
        result, diffs = _check_data_integrity(df1, df2, sample_size=1000)
        assert result["passed"] is False
        assert "error" in result["detail"].lower()


# ─── Aggregate Parity Check Tests (mocked) ───────────────────────────────────

class TestAggregateParity:

    def test_no_numeric_columns(self):
        df = _make_mock_df([("name", T.StringType())])
        result = _check_aggregate_parity(df, df, sample_size=1000, tolerance=0.0001)
        assert result["passed"] is True
        assert "No numeric" in result["detail"]

    def test_matching_aggregates(self):
        fields = [("id", T.IntegerType()), ("amount", T.DoubleType())]
        rows = [{"id": 1, "amount": 10.0}, {"id": 2, "amount": 20.0}]
        df1 = _make_mock_df(fields, rows=rows)
        df2 = _make_mock_df(fields, rows=rows)

        # Both should return same sums.
        # Dict-based agg produces columns named "sum(col)", e.g. "sum(id)".
        agg_row = MagicMock()
        agg_row.__getitem__ = MagicMock(side_effect=lambda k: 3 if k == "sum(id)" else 30.0)
        df1.limit.return_value.agg.return_value.collect.return_value = [agg_row]
        df2.limit.return_value.agg.return_value.collect.return_value = [agg_row]

        result = _check_aggregate_parity(df1, df2, sample_size=1000, tolerance=0.0001)
        assert result["passed"] is True

    def test_mismatched_aggregates(self):
        fields = [("id", T.IntegerType()), ("amount", T.DoubleType())]
        df1 = _make_mock_df(fields)
        df2 = _make_mock_df(fields)

        agg1 = MagicMock()
        agg1.__getitem__ = MagicMock(side_effect=lambda k: 3 if k == "sum(id)" else 30.0)
        df1.limit.return_value.agg.return_value.collect.return_value = [agg1]

        agg2 = MagicMock()
        agg2.__getitem__ = MagicMock(side_effect=lambda k: 3 if k == "sum(id)" else 999.0)
        df2.limit.return_value.agg.return_value.collect.return_value = [agg2]

        result = _check_aggregate_parity(df1, df2, sample_size=1000, tolerance=0.0001)
        assert result["passed"] is False
        assert "amount" in result["detail"]

    def test_aggregate_error_handled(self):
        df1 = MagicMock()
        type(df1).schema = PropertyMock(side_effect=RuntimeError("boom"))
        df2 = MagicMock()
        result = _check_aggregate_parity(df1, df2, sample_size=1000, tolerance=0.0001)
        assert result["passed"] is False
        assert "error" in result["detail"].lower()

    def test_decimal_type_matching(self):
        """DecimalType columns are compared with Decimal arithmetic, not float."""
        from decimal import Decimal
        fields = [("price", T.DecimalType(18, 6))]
        df1 = _make_mock_df(fields)
        df2 = _make_mock_df(fields)

        agg1 = MagicMock()
        agg1.__getitem__ = MagicMock(return_value=Decimal("123456789.123456"))
        df1.limit.return_value.agg.return_value.collect.return_value = [agg1]

        agg2 = MagicMock()
        agg2.__getitem__ = MagicMock(return_value=Decimal("123456789.123456"))
        df2.limit.return_value.agg.return_value.collect.return_value = [agg2]

        result = _check_aggregate_parity(df1, df2, sample_size=1000, tolerance=0.0001)
        assert result["passed"] is True

    def test_decimal_type_mismatch(self):
        """DecimalType mismatch is detected without float precision loss."""
        from decimal import Decimal
        fields = [("price", T.DecimalType(18, 6))]
        df1 = _make_mock_df(fields)
        df2 = _make_mock_df(fields)

        agg1 = MagicMock()
        agg1.__getitem__ = MagicMock(return_value=Decimal("100.000000"))
        df1.limit.return_value.agg.return_value.collect.return_value = [agg1]

        agg2 = MagicMock()
        agg2.__getitem__ = MagicMock(return_value=Decimal("200.000000"))
        df2.limit.return_value.agg.return_value.collect.return_value = [agg2]

        result = _check_aggregate_parity(df1, df2, sample_size=1000, tolerance=0.0001)
        assert result["passed"] is False
        assert "price" in result["detail"]


# ─── Full Validation Integration Tests (mocked) ─────────────────────────────

class TestValidateOptimization:

    def test_identical_dataframes_pass(self):
        fields = [("id", T.IntegerType()), ("name", T.StringType())]
        df = _make_mock_df(fields, row_count=3)

        # Aggregate parity: no numeric columns (StringType is not numeric)
        # Actually IntegerType IS numeric — need to set up agg mock
        agg_row = MagicMock()
        agg_row.__getitem__ = MagicMock(return_value=6)  # sum of ids
        df.limit.return_value.agg.return_value.collect.return_value = [agg_row]

        result = validate_optimization(df, df)
        assert result.passed is True
        assert result.confidence == "HIGH"
        assert result.confidence_score == 1.0
        assert len(result.checks) == 4

    def test_schema_mismatch_fails_fast(self):
        df1 = _make_mock_df([("id", T.IntegerType()), ("name", T.StringType())])
        df2 = _make_mock_df([("id", T.IntegerType())])
        result = validate_optimization(df1, df2)
        assert result.passed is False
        assert result.confidence == "LOW"
        # Should short-circuit — only schema check ran
        assert len(result.checks) == 1
        assert result.checks[0]["name"] == "schema_match"

    def test_row_count_mismatch_fails_fast(self):
        fields = [("id", T.IntegerType())]
        df1 = _make_mock_df(fields, row_count=100)
        df2 = _make_mock_df(fields, row_count=50)
        result = validate_optimization(df1, df2)
        assert result.passed is False
        # Schema passed, row count failed
        assert len(result.checks) == 2
        assert result.checks[1]["name"] == "row_count"

    def test_retry_degrades_confidence(self):
        fields = [("name", T.StringType())]
        df = _make_mock_df(fields, row_count=1)
        r1 = validate_optimization(df, df, attempt_number=1)
        r2 = validate_optimization(df, df, attempt_number=2)
        r3 = validate_optimization(df, df, attempt_number=3)
        assert r1.confidence_score > r2.confidence_score > r3.confidence_score

    def test_empty_dataframes(self):
        fields = [("id", T.IntegerType())]
        df = _make_mock_df(fields, row_count=0)
        result = validate_optimization(df, df)
        assert result.passed is True


# ─── Failure Prompt Builder Tests ────────────────────────────────────────────

class TestBuildFailurePrompt:

    def test_contains_failure_details(self):
        result = ValidationResult(
            passed=False,
            checks=[
                {"name": "schema_match", "passed": True, "detail": "ok"},
                {"name": "row_count", "passed": False, "detail": "original=100, optimized=50"},
            ],
            confidence="LOW",
            confidence_score=0.0,
        )
        prompt = build_failure_prompt(result, "optimized_df = df.filter(F.col('x') > 0)")
        assert "FAILED" in prompt
        assert "row_count" in prompt
        assert "optimized_df = df.filter" in prompt
        assert "Instructions" in prompt

    def test_contains_sample_diffs(self):
        result = ValidationResult(
            passed=False,
            checks=[
                {"name": "data_integrity", "passed": False, "detail": "3 differing rows"},
            ],
            sample_diffs={
                "only_in_original": [{"id": 1, "val": 10}],
                "only_in_optimized": [{"id": 1, "val": 99}],
            },
        )
        prompt = build_failure_prompt(result, "code")
        assert "Sample Differences" in prompt

    def test_contains_plan_diagnostics(self):
        result = ValidationResult(
            passed=False,
            checks=[{"name": "schema_match", "passed": False, "detail": "mismatch"}],
        )
        diags = [
            {"rule_id": "MISSING_BROADCAST", "message": "SortMergeJoin detected"},
        ]
        prompt = build_failure_prompt(result, "code", plan_diagnostics=diags)
        assert "MISSING_BROADCAST" in prompt
        assert "SortMergeJoin" in prompt

    def test_no_plan_diagnostics(self):
        result = ValidationResult(
            passed=False,
            checks=[{"name": "schema_match", "passed": False, "detail": "mismatch"}],
        )
        prompt = build_failure_prompt(result, "code", plan_diagnostics=None)
        assert "Plan Diagnostics" not in prompt
