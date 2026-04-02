import pytest
from optispark.safety import validate_safety

def test_safety_safe_code(spark):
    # Simple non-dangerous code
    code = "df_opt = df.withColumn('x', F.lit(1))"
    df = spark.range(10)
    
    is_safe, msg = validate_safety(code, df)
    
    assert is_safe is True
    assert "No high-memory operations detected" in msg

def test_safety_dangerous_no_df():
    code = "df_opt = df.withColumn('e', F.explode(F.array(1, 2)))"
    is_safe, msg = validate_safety(code, None)
    assert is_safe is False
    assert "no target_df provided" in msg

def test_safety_dangerous_with_safe_size(spark):
    code = "df_opt = df.withColumn('e', F.explode(F.array(1, 2)))"
    # A very tiny dataframe (safe size)
    df = spark.range(1)
    is_safe, msg = validate_safety(code, df, max_safe_size_mb=50)
    assert is_safe is True
    assert "passed safety thresholds" in msg

def test_safety_dangerous_with_unsafe_size(spark):
    code = "df_opt = df.withColumn('salt_array', F.explode(F.array(1, 2)))"
    # Force catalyst stats error or threshold breach by setting strict threshold
    df = spark.range(1000)
    # threshold extremely low to trigger the false condition
    is_safe, msg = validate_safety(code, df, max_safe_size_mb=-1.0)
    assert is_safe is False
    assert "Risk of OOM" in msg

def test_safety_catalyst_stats_error():
    # Pass a dummy object instead of real PySpark DataFrame to trigger Exception in Catalyst try block
    class DummyDF:
        @property
        def _jdf(self):
            raise ValueError("Fake Catalyst Error")
            
    code = "df_opt = df.withColumn('e', F.explode(F.array(1, 2)))"
    is_safe, msg = validate_safety(code, DummyDF())
    assert is_safe is False
    assert "Failed to calculate Catalyst stats: Fake Catalyst Error" in msg
