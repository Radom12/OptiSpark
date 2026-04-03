import pytest
from optispark.parser import extract_features_from_system_tables, _calculate_bottlenecks, extract_features_from_logs

def test_calculate_bottlenecks():
    metrics = {
        1: { # Healthy
            "times": [100, 110, 105],
            "mem_spill": 0, "disk_spill": 0,
            "records_read": 100, "records_written": 100
        },
        2: { # Skewed, Spill, Exploding
            "times": [10, 15, 500], # max is 500, median is 15 -> skew is 33.3
            "mem_spill": 1024*1024*5, # 5MB
            "disk_spill": 0,
            "records_read": 10, "records_written": 20000 # amp factor > 1000
        },
        3: { # Small files
            "times": [5] * 15000, # median < 100, tasks > 10000
            "mem_spill": 0, "disk_spill": 0,
            "records_read": 0, "records_written": 0
        },
        4: { # Skipped (too few tasks)
            "times": [100],
            "mem_spill": 0, "disk_spill": 0,
            "records_read": 0, "records_written": 0
        },
        5: { # 0 median exception avoid
            "times": [0, 0, 0],
            "mem_spill": 0, "disk_spill": 0,
            "records_read": 0, "records_written": 0
        }
    }
    
    features = _calculate_bottlenecks(metrics)
    assert len(features) == 3 # stage 4 and 5 skipped
    
    health_feats = next(f for f in features if f["stage_id"] == 1)
    assert not health_feats["flags"]
    
    skew_feats = next(f for f in features if f["stage_id"] == 2)
    assert "CRITICAL_SKEW" in skew_feats["flags"]
    assert "SKEW" in skew_feats["flags"]
    assert "SHUFFLE_SPILL" in skew_feats["flags"]
    assert "EXPLODING_JOIN" in skew_feats["flags"]
    
    small_feats = next(f for f in features if f["stage_id"] == 3)
    assert "SMALL_FILES" in small_feats["flags"]

def test_extract_features_from_system_tables_success():
    from unittest.mock import MagicMock
    # Create a fully mocked spark session that returns predictable data
    mock_spark = MagicMock()
    mock_row = {"total_duration_ms": 1000, "avg_task_duration": 100, "max_task_duration_ms": 500}
    mock_spark.sql.return_value.collect.return_value = [mock_row]

    res = extract_features_from_system_tables(mock_spark, "query123")
    assert len(res) == 1
    assert res[0]["stage_id"] == "ServerlessQuery"
    assert res[0]["skew_ratio"] == 5.0  # 500 / 100

def test_extract_features_from_system_tables_empty(spark):
    assert extract_features_from_system_tables(None, "query123") is None
    assert extract_features_from_system_tables(spark, None) is None

def test_extract_features_from_system_tables_fail(spark):
    # force exception
    original_sql = spark.sql
    try:
        spark.sql = lambda x: 1/0
        assert extract_features_from_system_tables(spark, "q1") is None
    finally:
        spark.sql = original_sql

def test_extract_features_from_logs_empty():
    assert extract_features_from_logs("/invalid/path") is None

import json
def test_extract_features_from_logs_success(tmp_path):
    # simulate log dir
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    app_dir = log_dir / "app-123"
    app_dir.mkdir()
    event_file = app_dir / "events_1"
    
    event = {
        "Event": "SparkListenerTaskEnd",
        "Task End Reason": {"Reason": "Success"},
        "Stage ID": 1,
        "Task Metrics": {
            "Executor Run Time": 100,
            "Memory Bytes Spilled": 0,
            "Disk Bytes Spilled": 0,
            "Input Metrics": {"Records Read": 10},
            "Output Metrics": {"Records Written": 10},
            "Shuffle Write Metrics": {"Shuffle Records Written": 0}
        }
    }
    # write two events so it calculates median instead of skipping
    event_file.write_text(json.dumps(event) + "\n" + json.dumps(event))
    
    features = extract_features_from_logs(str(log_dir))
    assert len(features) == 1
    assert features[0]["stage_id"] == 1
