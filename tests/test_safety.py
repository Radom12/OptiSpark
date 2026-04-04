import pytest
from unittest.mock import MagicMock
from optispark.safety import validate_safety, ReadOnlyValidator, secure_exec


def test_safety_safe_code():
    """Safe code with no dangerous operations."""
    code = "optimized_df = df.withColumn('x', F.lit(1))"
    mock_df = MagicMock()

    is_safe, msg = validate_safety(code, mock_df)
    assert is_safe is True
    assert "No high-memory operations detected" in msg


def test_safety_dangerous_no_df():
    """Dangerous code without target_df should fail."""
    code = "optimized_df = df.withColumn('e', F.explode(F.array(1, 2)))"
    is_safe, msg = validate_safety(code, None)
    assert is_safe is False
    assert "no target_df provided" in msg


def test_safety_dangerous_with_safe_size():
    """Dangerous code but DataFrame is small enough."""
    code = "optimized_df = df.withColumn('e', F.explode(F.array(1, 2)))"
    mock_df = MagicMock()
    # Chain: _jdf.queryExecution().optimizedPlan().stats().sizeInBytes() -> 1024 bytes (1 KB)
    mock_df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes.return_value = 1024

    is_safe, msg = validate_safety(code, mock_df, max_safe_size_mb=50)
    assert is_safe is True
    assert "passed safety thresholds" in msg


def test_safety_dangerous_with_unsafe_size():
    """Dangerous code with a large DataFrame should be blocked."""
    code = "optimized_df = df.withColumn('salt_array', F.explode(F.array(1, 2)))"
    mock_df = MagicMock()
    # 100 MB
    mock_df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes.return_value = 100 * 1024 * 1024

    is_safe, msg = validate_safety(code, mock_df, max_safe_size_mb=50)
    assert is_safe is False
    assert "Risk of OOM" in msg


def test_safety_catalyst_stats_error():
    """When Catalyst stats can't be computed, should fail safely."""
    class DummyDF:
        @property
        def _jdf(self):
            raise ValueError("Fake Catalyst Error")

    code = "optimized_df = df.withColumn('e', F.explode(F.array(1, 2)))"
    is_safe, msg = validate_safety(code, DummyDF())
    assert is_safe is False
    assert "Failed to calculate Catalyst stats: Fake Catalyst Error" in msg


# --- ReadOnlyValidator tests ---

def test_validator_allows_safe_code():
    """ReadOnlyValidator should not raise on safe transformations."""
    import ast
    code = "result = df.select('a').filter(df.b > 0).groupBy('a').count()"
    tree = ast.parse(code)
    ReadOnlyValidator().visit(tree)  # should not raise


def test_validator_blocks_save():
    """ReadOnlyValidator should block .save() calls."""
    import ast
    code = "df.write.mode('overwrite').save('/some/path')"
    tree = ast.parse(code)
    with pytest.raises(ValueError, match="blocked"):
        ReadOnlyValidator().visit(tree)


def test_validator_blocks_write_attribute():
    """ReadOnlyValidator should block .write attribute access (e.g. df.write.parquet)."""
    import ast
    code = "df.write.parquet('/some/path')"
    tree = ast.parse(code)
    with pytest.raises(ValueError, match="blocked access to '.write' attribute"):
        ReadOnlyValidator().visit(tree)


def test_validator_blocks_save_as_table():
    """ReadOnlyValidator should block .saveAsTable()."""
    import ast
    code = "df.write.mode('overwrite').saveAsTable('my_table')"
    tree = ast.parse(code)
    with pytest.raises(ValueError):
        ReadOnlyValidator().visit(tree)


def test_validator_blocks_insert_into():
    """ReadOnlyValidator should block .insertInto()."""
    import ast
    code = "df.write.insertInto('my_table')"
    tree = ast.parse(code)
    with pytest.raises(ValueError):
        ReadOnlyValidator().visit(tree)


def test_validator_blocks_drop():
    """ReadOnlyValidator blocks .drop() method calls."""
    import ast
    code = "df.drop()"
    tree = ast.parse(code)
    with pytest.raises(ValueError, match="blocked destructive method 'drop'"):
        ReadOnlyValidator().visit(tree)


def test_validator_blocks_destructive_sql():
    """ReadOnlyValidator should block destructive SQL via spark.sql()."""
    import ast
    code = "spark.sql('DROP TABLE my_table')"
    tree = ast.parse(code)
    with pytest.raises(ValueError, match="blocked destructive SQL command"):
        ReadOnlyValidator().visit(tree)


def test_validator_allows_select_sql():
    """ReadOnlyValidator should allow read-only SQL via spark.sql()."""
    import ast
    code = "result = spark.sql('SELECT * FROM my_table WHERE id = 1')"
    tree = ast.parse(code)
    ReadOnlyValidator().visit(tree)  # should not raise


# --- secure_exec tests ---

def test_secure_exec_runs_safe_code():
    """secure_exec should execute safe code and update local_vars."""
    local_vars = {}
    secure_exec("x = 1 + 1", {}, local_vars)
    assert local_vars["x"] == 2


def test_secure_exec_blocks_unsafe_code():
    """secure_exec should raise before executing unsafe code."""
    with pytest.raises(ValueError, match="blocked"):
        secure_exec("df.write.parquet('/path')", {}, {})
