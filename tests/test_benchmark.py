import pytest
from unittest.mock import MagicMock, patch
from optispark.benchmark import run_benchmark


def _make_mock_df(row_count=10):
    """Create a mock DataFrame that works with benchmark's limit/count/cache flow."""
    mock_df = MagicMock()
    mock_df.sparkSession = MagicMock()
    mock_df.count.return_value = row_count * 1000  # Total rows (if needed)

    # limit().cache() returns a sampled df
    sampled = MagicMock()
    sampled.count.return_value = row_count
    mock_df.limit.return_value.cache.return_value = sampled

    return mock_df, sampled


def test_benchmark_success():
    mock_df, sampled = _make_mock_df(row_count=10)

    # Use simple assignment — works in exec sandbox with mocks
    code = "optimized_df = df"
    res = run_benchmark(mock_df, code)

    assert res["status"] == "success"
    assert "original_time_sec" in res
    assert "fixed_time_sec" in res
    assert "improvement_pct" in res


def test_benchmark_empty_df():
    mock_df, sampled = _make_mock_df(row_count=0)

    res = run_benchmark(mock_df, "optimized_df = df")
    assert res["status"] == "error"
    assert "Sampled DataFrame is empty" in res["message"]


def test_benchmark_execution_error():
    mock_df, sampled = _make_mock_df(row_count=10)

    # Code that will raise during exec
    code = "optimized_df = this_function_does_not_exist()"
    res = run_benchmark(mock_df, code)

    assert res["status"] == "error"
    assert "AI Code Execution Failed" in res["message"]


def test_benchmark_no_df_opt():
    mock_df, sampled = _make_mock_df(row_count=10)

    # Code that runs but doesn't assign to optimized_df
    code = "some_other_df = df"
    res = run_benchmark(mock_df, code)

    assert res["status"] == "error"
    assert "not assign the result to 'optimized_df'" in res["message"]


def test_benchmark_total_failure():
    # Pass None to force the outer exception
    res = run_benchmark(None, "optimized_df = None")
    assert res["status"] == "error"
    assert "Benchmark failed" in res["message"]
