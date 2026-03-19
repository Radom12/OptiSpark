from pyspark.sql import SparkSession
import statistics

class OptiSparkListener:
    def __init__(self):
        self.stage_metrics = {}

    def onTaskEnd(self, taskEnd):
        # Capture metrics only if the task was successful
        if taskEnd.reason().toString() == "Success":
            stage_id = taskEnd.stageId()
            # Task metrics are in milliseconds
            run_time = taskEnd.taskMetrics().executorRunTime()
            
            if stage_id not in self.stage_metrics:
                self.stage_metrics[stage_id] = []
            self.stage_metrics[stage_id].append(run_time)

    def get_features(self):
        features = []
        for stage_id, times in self.stage_metrics.items():
            if len(times) > 1 and statistics.median(times) > 0:
                features.append({
                    "stage_id": stage_id,
                    "skew_ratio": round(max(times) / statistics.median(times), 2)
                })
        return features