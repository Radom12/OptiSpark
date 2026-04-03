import pytest
from optispark.listener import OptiSparkListener

class MockTaskMetrics:
    def __init__(self, run_time, mem_spill, disk_spill, read, write):
        self._run_time = run_time
        self._mem_spill = mem_spill
        self._disk_spill = disk_spill
        self._read = read
        self._write = write
        
    def executorRunTime(self): return self._run_time
    def memoryBytesSpilled(self): return self._mem_spill
    def diskBytesSpilled(self): return self._disk_spill
    
    def inputMetrics(self):
        class IM:
            def recordsRead(s): return self._read
        return IM() if self._read > 0 else None
        
    def outputMetrics(self):
        class OM:
            def recordsWritten(s): return self._write
        return OM() if self._write > 0 else None
        
    def shuffleWriteMetrics(self): return None

class MockTaskEnd:
    def __init__(self, reason, stage_id, metrics):
        self._reason = reason
        self._stage_id = stage_id
        self._metrics = metrics
        
    def reason(self):
        class Reason:
            def toString(s): return self._reason
        return Reason()
        
    def stageId(self): return self._stage_id
    def taskMetrics(self): return self._metrics

def test_listener():
    listener = OptiSparkListener()
    
    task_end1 = MockTaskEnd(
        "Success", 
        stage_id=1, 
        metrics=MockTaskMetrics(run_time=100, mem_spill=1024, disk_spill=0, read=10, write=20)
    )
    
    # Send event
    listener.onTaskEnd(task_end1)
    
    assert 1 in listener.stage_metrics
    assert listener.stage_metrics[1]["times"] == [100]
    assert listener.stage_metrics[1]["mem_spill"] == 1024
    assert listener.stage_metrics[1]["disk_spill"] == 0
    assert listener.stage_metrics[1]["records_read"] == 10
    assert listener.stage_metrics[1]["records_written"] == 20
    
    # Send another task for same stage to test aggregation
    task_end2 = MockTaskEnd(
        "Success", 
        stage_id=1, 
        metrics=MockTaskMetrics(run_time=50, mem_spill=0, disk_spill=512, read=0, write=0)
    )
    listener.onTaskEnd(task_end2)
    
    assert len(listener.stage_metrics[1]["times"]) == 2
    assert listener.stage_metrics[1]["mem_spill"] == 1024
    assert listener.stage_metrics[1]["disk_spill"] == 512
    assert listener.stage_metrics[1]["records_read"] == 10
    
    # Failed task should be ignored
    failed_task = MockTaskEnd("Failed", stage_id=2, metrics=MockTaskMetrics(10,0,0,0,0))
    listener.onTaskEnd(failed_task)
    assert 2 not in listener.stage_metrics
