from unittest.mock import MagicMock

def _make_mock_df(row_count=10):
    mock_df = MagicMock()
    mock_df.sparkSession = MagicMock()
    mock_df.count.return_value = row_count * 1000

    sampled = MagicMock()
    sampled.count.return_value = row_count
    mock_df.limit.return_value.cache.return_value = sampled

    return mock_df, sampled

mock_df, sampled = _make_mock_df(10)
df_sampled = mock_df.limit(100).cache()
baseline_rows = df_sampled.count()
print(repr(baseline_rows))
