import time
import traceback
import pyspark.sql.functions as F
from pyspark.sql import Window

def run_benchmark(original_df, generated_code_str):
    """
    Executes a sampled 1% dry-run of both the original DataFrame and the AI-generated
    fixed DataFrame to benchmark and compare their execution times.
    """
    try:
        spark = original_df.sparkSession
        
        # 1. Create a 1% Sample and cache it to avoid I/O skewing the CPU/Shuffle benchmark
        print("  ⏳ Preparing 1% data sample for benchmark...", end="")
        df_sampled = original_df.sample(fraction=0.01, seed=42).cache()
        # Force materialization and caching before timing
        baseline_rows = df_sampled.count()
        print(f" ✔ (Cached {baseline_rows:,} rows)")
        
        if baseline_rows == 0:
            return {"status": "error", "message": "Sampled DataFrame is empty. Cannot run benchmark."}

        # 2. Benchmark Original Execution
        print("  ⏳ Benchmarking original DataFrame...", end="")
        start_orig = time.perf_counter()
        _ = df_sampled.count()  # Trigger action
        end_orig = time.perf_counter()
        original_time_sec = end_orig - start_orig
        print(f" ✔ ({original_time_sec:.2f}s)")

        # 3. Setup Sandbox Execution Environment
        print("  ⏳ Executing AI optimization logic...", end="")
        local_env = {
            "df": df_sampled,
            "spark": spark,
            "F": F,
            "Window": Window
        }
        
        # 4. Execute the AI code
        # We wrap in try/except because LLM code might contain Catalyst errors
        try:
            exec(generated_code_str.strip(), {}, local_env)
        except Exception as exec_err:
            print(f" ✖ Failed")
            return {"status": "error", "message": f"AI Code Execution Failed: {str(exec_err)}"}
            
        if "df_opt" not in local_env:
            print(f" ✖ Failed")
            return {"status": "error", "message": "The generated code did not assign the result to 'df_opt'."}
            
        df_opt = local_env["df_opt"]
        print(" ✔")
        
        # 5. Benchmark Fixed Execution
        print("  ⏳ Benchmarking fixed DataFrame...", end="")
        start_fixed = time.perf_counter()
        # Trigger an action on the fixed dataframe to force Physical Plan evaluation
        _ = df_opt.count()
        end_fixed = time.perf_counter()
        fixed_time_sec = end_fixed - start_fixed
        print(f" ✔ ({fixed_time_sec:.2f}s)")
        
        # 6. Calculate Improvement
        if original_time_sec > 0:
            improvement_pct = ((original_time_sec - fixed_time_sec) / original_time_sec) * 100
        else:
            improvement_pct = 0.0

        return {
            "status": "success",
            "original_time_sec": round(original_time_sec, 3),
            "fixed_time_sec": round(fixed_time_sec, 3),
            "improvement_pct": round(improvement_pct, 2)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Benchmark failed: {str(e)}\n{traceback.format_exc()}"}
