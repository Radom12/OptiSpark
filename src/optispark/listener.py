from pyspark.sql import SparkSession
import statistics

class OptiSparkListener:
    def __init__(self):
        # Dictionary structure: {stage_id: {"times": [], "mem_spill": 0, "disk_spill": 0, "records_read": 0, "records_written": 0}}
        self.stage_metrics = {}

    def onTaskEnd(self, taskEnd):
        # Capture metrics only if the task was successful
        if taskEnd.reason().toString() == "Success":
            stage_id = taskEnd.stageId()
            metrics = taskEnd.taskMetrics()
            
            run_time = metrics.executorRunTime()
            mem_spill = metrics.memoryBytesSpilled()
            disk_spill = metrics.diskBytesSpilled()
            
            # Input metrics
            records_read = metrics.inputMetrics().recordsRead() if metrics.inputMetrics() else 0
            
            # Output metrics (Shuffle output or direct output)
            records_written = 0
            if metrics.outputMetrics():
                records_written += metrics.outputMetrics().recordsWritten()
            if metrics.shuffleWriteMetrics():
                records_written += metrics.shuffleWriteMetrics().recordsWritten()
            
            if stage_id not in self.stage_metrics:
                self.stage_metrics[stage_id] = {
                    "times": [],
                    "mem_spill": 0,
                    "disk_spill": 0,
                    "records_read": 0,
                    "records_written": 0
                }
                
            self.stage_metrics[stage_id]["times"].append(run_time)
            self.stage_metrics[stage_id]["mem_spill"] += mem_spill
            self.stage_metrics[stage_id]["disk_spill"] += disk_spill
            self.stage_metrics[stage_id]["records_read"] += records_read
            self.stage_metrics[stage_id]["records_written"] += records_written