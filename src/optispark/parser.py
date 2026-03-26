import os
import json
import statistics
import zstandard as zstd
import io

def extract_features_from_logs(log_dir):
    """Attempt to extract features from physical Spark Event Logs."""
    if not log_dir or not os.path.exists(log_dir):
        return None
    
    log_items = [f for f in os.listdir(log_dir) if not f.startswith('.')]
    if not log_items:
        return None
        
    latest_log = max(log_items, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
    log_path = os.path.join(log_dir, latest_log)
    
    files_to_read = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.startswith("events_")] if os.path.isdir(log_path) else [log_path]
    
    stage_metrics = {}
    for file_path in files_to_read:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            f.seek(0)
            text_stream = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(f), encoding='utf-8') if magic == b'\x28\xb5\x2f\xfd' else io.TextIOWrapper(f, encoding='utf-8')
            for line in text_stream:
                try:
                    event = json.loads(line)
                    if event.get("Event") == "SparkListenerTaskEnd" and event.get("Task End Reason", {}).get("Reason") == "Success":
                        stage_id = event["Stage ID"]
                        metrics = event.get("Task Metrics", {})
                        
                        if stage_id not in stage_metrics:
                            stage_metrics[stage_id] = {
                                "times": [], "mem_spill": 0, "disk_spill": 0, 
                                "records_read": 0, "records_written": 0
                            }
                        
                        sm = stage_metrics[stage_id]
                        sm["times"].append(metrics.get("Executor Run Time", 0))
                        sm["mem_spill"] += metrics.get("Memory Bytes Spilled", 0)
                        sm["disk_spill"] += metrics.get("Disk Bytes Spilled", 0)
                        sm["records_read"] += metrics.get("Input Metrics", {}).get("Records Read", 0)
                        
                        records_written = 0
                        records_written += metrics.get("Output Metrics", {}).get("Records Written", 0)
                        records_written += metrics.get("Shuffle Write Metrics", {}).get("Shuffle Records Written", 0)
                        sm["records_written"] += records_written
                        
                except json.JSONDecodeError: continue

    return _calculate_bottlenecks(stage_metrics)

def extract_features_from_system_tables(spark, query_id):
    """Fallback: Extract metrics from Databricks system.query.history."""
    if not spark or not query_id:
        return None
        
    print(f"🔄 Fallback: Extracting Serverless metrics for Query ID: {query_id}...")
    
    query = f"""
        SELECT 
            total_duration_ms,
            (total_task_duration_ms / NULLIF(total_tasks, 0)) as avg_task_duration,
            max_task_duration_ms
        FROM system.query.history 
        WHERE query_id = '{query_id}'
    """
    try:
        row = spark.sql(query).collect()[0]
        max_t = row['max_task_duration_ms']
        avg_t = row['avg_task_duration']
        skew = round(max_t / avg_t, 2) if avg_t and avg_t > 0 else 1.0
        
        return [{"stage_id": "ServerlessQuery", "skew_ratio": skew}]
    except Exception as e:
        print(f"❌ Serverless fallback failed: {str(e)}")
        return None

def _calculate_bottlenecks(stage_metrics):
    """
    Analyzes aggregated stage metrics to detect PySpark performance bottlenecks.
    Calculates Skew, Spill, Small Files parsing, and Amplification factors.
    """
    features = []
    for stage_id, data in stage_metrics.items():
        times = data["times"]
        num_tasks = len(times)
        
        if num_tasks <= 1:
            continue
            
        median_time = statistics.median(times)
        if median_time == 0:
            continue
            
        # 1. Skew Ratio: Max Time vs Median Time
        skew_ratio = round(max(times) / median_time, 2)
        
        # 2. Shuffle Spill (Memory & Disk in MBs)
        disk_spill_mb = round(data["disk_spill"] / (1024 * 1024), 2)
        mem_spill_mb = round(data["mem_spill"] / (1024 * 1024), 2)
        
        # 3. Exploding Joins (Amplification Ratio)
        amp_factor = 1.0
        if data["records_read"] > 0:
            amp_factor = round(data["records_written"] / data["records_read"], 2)
            
        # Compile Bottleneck Flags
        flags = []
        if skew_ratio > 3.0:
            flags.append("SKEW")
        if skew_ratio > 5.0:
            flags.append("CRITICAL_SKEW")
        
        # Threshold: If any spill occurred, it's causing disk fallback.
        if disk_spill_mb > 0 or mem_spill_mb > 0:
            flags.append("SHUFFLE_SPILL")
            
        # Threshold: > 10,000 tasks that each run for under 100ms indicates serious over-partitioning
        if median_time < 100 and num_tasks > 10000:
            flags.append("SMALL_FILES")
            
        # Threshold: If a stage outputs 1000x more records than it read, it's likely a bad join state
        if amp_factor > 1000:
            flags.append("EXPLODING_JOIN")
            
        features.append({
            "stage_id": stage_id,
            "skew_ratio": skew_ratio,
            "spilled_disk_mb": disk_spill_mb,
            "spilled_mem_mb": mem_spill_mb,
            "median_task_ms": median_time,
            "num_tasks": num_tasks,
            "amplification_factor": amp_factor,
            "flags": flags
        })
        
    return features