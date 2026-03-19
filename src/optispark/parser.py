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
                        stage_metrics.setdefault(stage_id, []).append(event.get("Task Metrics", {}).get("Executor Run Time", 0))
                except json.JSONDecodeError: continue

    return _calculate_skew(stage_metrics)

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

def _calculate_skew(stage_metrics):
    features = []
    for stage_id, times in stage_metrics.items():
        if len(times) > 1 and statistics.median(times) > 0:
            features.append({
                "stage_id": stage_id,
                "skew_ratio": round(max(times) / statistics.median(times), 2)
            })
    return features