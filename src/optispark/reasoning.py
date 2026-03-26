import json
from google import genai
from google.genai import types

class ReasoningEngine:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3-flash-preview"
        self.system_instruction = """
            You are OptiSpark, an interactive PySpark performance architect. 
            You have access to the user's Spark DAG metrics and the exact PySpark code they executed.
            When the user asks for a fix, map their request to the provided metrics and code.
            Provide explanations and exact PySpark code rewrites.
        """

    def start_chat(self, context_metrics, code_context=None):
        """Initializes a chat session loaded with the Spark DAG and Code context."""
        
        # We start the chat with the system instructions
        chat = self.client.chats.create(
            model=self.model_id,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                # We do NOT force JSON here, because we want a natural conversation
            )
        )
        
        # Inject the invisible context prompt to ground the LLM
        system_context = f"""
        [SYSTEM INJECTION - DO NOT REPLY TO THIS]
        Here is the current execution context for this session:
        Metrics: {json.dumps(context_metrics)}
        Code Executed: {code_context if code_context else 'No code context available.'}
        Wait for the user's first question.
        """
        chat.send_message(system_context)
        
        return chat