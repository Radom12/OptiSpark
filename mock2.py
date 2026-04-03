import traceback
from unittest.mock import MagicMock
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

mock_df, sampled = _make_mock_df(row_count=10)
res = run_benchmark(mock_df, "df_opt = df")
print(res)
