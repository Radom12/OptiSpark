import json
from google import genai
from google.genai import types

class ReasoningEngine:
    def __init__(self, api_key):
        # The new SDK uses a Client object
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3-flash-preview"
        self.system_instruction = """
            You are OptiSpark, an expert PySpark performance architect. 
            Analyze structured stage metrics, identify root causes, and provide PySpark code fixes.
            Respond ONLY in JSON with keys: "diagnosis", "explanation", "pyspark_code".
        """

    def get_recommendation(self, anomaly):
        print(f"🧠 [2/4] Analyzing Anomaly in Stage {anomaly['stage_id']} (Skew: {anomaly['skew_ratio']})...")
        
        # New syntax for structured generation
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=f"Analyze this stage:\n{json.dumps(anomaly)}",
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)