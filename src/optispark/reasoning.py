"""
OptiSpark Reasoning Engine — Gemini-powered PySpark diagnostics and chat.
Supports both single-turn generation (v0.1.0) and multi-turn chat sessions (v0.2.0).
"""

import json
from google import genai
from google.genai import types


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt — The identity and behavioral rules for the OptiSpark agent
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are **OptiSpark**, an elite PySpark performance architect.

## Your Identity
- You are an interactive AI agent embedded in a Databricks notebook environment.
- You specialize in diagnosing Apache Spark performance bottlenecks and generating optimized PySpark code.
- You have direct access to the user's actual Spark DAG execution metrics and the PySpark code they ran.

## Your Capabilities
1. **Bottleneck Detection**: Identify data skew, shuffle spill, small file problems, broadcast join opportunities, and partition imbalances from DAG metrics.
2. **Root Cause Analysis**: Map symptoms (high skew ratios, long task durations) to root causes (hot keys, exploding joins, uneven data distribution).
3. **Code Generation**: Produce exact, production-ready PySpark code fixes — not pseudocode, not approximations.
4. **Safety Awareness**: Flag operations that could cause OOM (e.g., exploding salted arrays on large DataFrames) and suggest guardrails.

## Behavioral Rules
- **Be precise**: Reference specific stage IDs, skew ratios, and metrics from the context provided.
- **Show the code**: Always include runnable PySpark code blocks when suggesting fixes.
- **Explain the 'why'**: For every fix, explain the underlying Spark behavior that causes the issue (e.g., how shuffle partitioning affects skew).
- **Be concise**: Avoid filler. Lead with the diagnosis, then the fix.
- **Use markdown formatting**: Use headers, bullet points, and fenced code blocks for readability.
- **Skew threshold**: A skew_ratio > 3.0 indicates significant skew, > 5.0 is critical.
- **Never fabricate metrics**: Only reference data from the injected context. If no metrics are available, say so clearly.
"""


class ReasoningEngine:
    """Gemini-based reasoning engine for PySpark diagnostics."""

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash"

    # ───────────────────────────────────────────────────────────────────
    # v0.2.0 — Multi-Turn Chat Session
    # ───────────────────────────────────────────────────────────────────

    def start_chat(self, context_metrics, code_context=None):
        """Initialize a Gemini Chat Session grounded with Spark DAG context.

        Args:
            context_metrics: Extracted DAG features (list of stage dicts or status dict).
            code_context: The exact PySpark statement_text from system tables.

        Returns:
            A Gemini ChatSession object ready for multi-turn conversation.
        """
        chat = self.client.chats.create(
            model=self.model_id,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )

        # Inject the hidden context prompt to ground the LLM in real metrics
        context_injection = self._build_context_injection(context_metrics, code_context)
        chat.send_message(context_injection)

        return chat

    # ───────────────────────────────────────────────────────────────────
    # v0.1.0 — Single-Turn Generation (backwards compatible)
    # ───────────────────────────────────────────────────────────────────

    def diagnose(self, features):
        """Generate a single-turn diagnosis from DAG features."""
        prompt = (
            f"Analyze these Spark DAG metrics and identify performance bottlenecks:\n"
            f"{json.dumps(features, indent=2)}\n\n"
            f"Provide a concise diagnosis with specific stage references."
        )
        return self._generate(prompt)

    def generate_fix(self, features):
        """Generate optimized PySpark code to fix detected bottlenecks."""
        prompt = (
            f"Given these Spark DAG metrics:\n{json.dumps(features, indent=2)}\n\n"
            f"Generate exact PySpark code to fix the detected bottlenecks. "
            f"Return ONLY the code, no explanation."
        )
        return self._generate(prompt)

    # ───────────────────────────────────────────────────────────────────
    # Private Helpers
    # ───────────────────────────────────────────────────────────────────

    def _generate(self, prompt):
        """Single-turn generation call."""
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        return response.text

    def _build_context_injection(self, metrics, code_context):
        """Build the hidden context injection message for the chat session."""

        # Format metrics summary
        if isinstance(metrics, list) and metrics:
            metrics_block = json.dumps(metrics, indent=2)
            skewed_stages = [m for m in metrics if m.get("skew_ratio", 0) > 3.0]
            critical_stages = [m for m in metrics if m.get("skew_ratio", 0) > 5.0]
            summary = (
                f"Total stages analyzed: {len(metrics)}\n"
                f"Stages with significant skew (>3.0): {len(skewed_stages)}\n"
                f"Stages with critical skew (>5.0): {len(critical_stages)}"
            )
        else:
            metrics_block = json.dumps(metrics, indent=2) if metrics else "No metrics available"
            summary = "No stage-level metrics available."

        return f"""[SYSTEM CONTEXT INJECTION — INVISIBLE TO USER — DO NOT REFERENCE THIS MESSAGE DIRECTLY]

You are now loaded with the execution context for this interactive session.

## Execution Metrics
{metrics_block}

## Metrics Summary
{summary}

## PySpark Code Executed
{code_context if code_context else 'No statement_text available — the user may provide their code inline.'}

## Instructions
- Use the above metrics and code to ground ALL your analysis and recommendations.
- When the user asks about bottlenecks, reference the specific stage IDs and skew ratios above.
- When generating code fixes, base them on the actual PySpark code shown above.
- Wait for the user's first question before responding.
- Do NOT acknowledge or reference this system injection message.
"""