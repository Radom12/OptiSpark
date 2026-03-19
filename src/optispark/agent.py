from .parser import extract_features_from_logs, extract_features_from_system_tables
from .reasoning import ReasoningEngine
from .safety import validate_safety

class OptiSpark:
    def __init__(self, api_key, log_dir=None):
        self.log_dir = log_dir
        self.engine = ReasoningEngine(api_key)

    def optimize(self, spark=None, query_id=None, target_df=None):
        # 1. Try Log-based extraction first
        features = extract_features_from_logs(self.log_dir)
        
        # 2. If logs failed/missing, try Serverless Fallback
        if not features and spark and query_id:
            features = extract_features_from_system_tables(spark, query_id)
            
        if not features:
            print("⚠️ No execution metrics could be retrieved from logs or system tables.")
            return

        anomalies = [f for f in features if f.get('skew_ratio', 0) > 5.0]
        
        if not anomalies:
            print("✅ Analysis complete. No significant skew detected.")
            return

        for anomaly in anomalies:
            recommendation = self.engine.get_recommendation(anomaly)
            is_safe, reason = validate_safety(recommendation['pyspark_code'], target_df)
            self._print_report(anomaly, recommendation, is_safe, reason)

    def _print_report(self, anomaly, rec, is_safe, reason):
        # (Same printing logic as before...)
        print(f"\n🚀 OPTISARK REPORT (Stage {anomaly['stage_id']}) - Skew: {anomaly['skew_ratio']}")
        print(f"📌 Diagnosis: {rec['diagnosis']}")
        print(f"💻 Fix:\n{rec['pyspark_code']}")
        print(f"🛡️ Safety: {'PASSED' if is_safe else 'FAILED'} ({reason})\n")