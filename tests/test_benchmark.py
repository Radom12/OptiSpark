import pytest
from optispark.benchmark import run_benchmark

def test_benchmark_success(spark):
    df = spark.range(100)
    # Simple valid optimization
    code = "df_opt = df.withColumn('sample_col', F.lit(1)).filter(F.col('id') < 50)"
    res = run_benchmark(df, code)
    
    assert res["status"] == "success"
    assert "original_time_sec" in res
    assert "fixed_time_sec" in res
    assert "improvement_pct" in res
    
def test_benchmark_empty_df(spark):
    df = spark.range(0) # Empty
    code = "df_opt = df"
    res = run_benchmark(df, code)
    assert res["status"] == "error"
    assert "Sampled DataFrame is empty" in res["message"]
    
def test_benchmark_execution_error(spark):
    df = spark.range(10)
    # Syntax error code
    code = "df_opt = df.withColumn('a', nonexistent_func())"
    res = run_benchmark(df, code)
    assert res["status"] == "error"
    assert "AI Code Execution Failed" in res["message"]

def test_benchmark_no_df_opt(spark):
    df = spark.range(10)
    code = "some_other_df = df.withColumn('x', F.lit(1))"
    res = run_benchmark(df, code)
    assert res["status"] == "error"
    assert "not assign the result to 'df_opt'" in res["message"]

def test_benchmark_total_failure():
    # Pass None to force an outer exception
    res = run_benchmark(None, "df_opt = None")
    assert res["status"] == "error"
    assert "Benchmark failed" in res["message"]
