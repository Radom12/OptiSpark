def validate_safety(code, target_df, max_safe_size_mb=50):
    """Validates AI-generated code against cluster safety constraints."""
    print("🛡️ [3/4] Running safety checks...")
    
    if "F.explode" in code or "salt_array" in code:
        if target_df is None:
            return False, "Dangerous operation detected, but no target_df provided for size validation."
            
        try:
            # Tap Catalyst Optimizer for metadata
            size_bytes = target_df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb > max_safe_size_mb:
                return False, f"Risk of OOM. Target DF is {size_mb:.2f} MB (Threshold: {max_safe_size_mb} MB)."
            return True, f"Code passed safety thresholds. DF size: {size_mb:.4f} MB."
        except Exception as e:
            return False, f"Failed to calculate Catalyst stats: {str(e)}"
            
    return True, "No high-memory operations detected in code."