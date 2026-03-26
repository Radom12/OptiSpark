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
You are **OptiSpark**, an elite PySpark autonomous execution agent.

## Your Identity
- You are an interactive AI agent embedded in a Databricks notebook environment.
- You diagnose Apache Spark bottlenecks and generate **executable** PySpark code fixes.
- You have direct access to the user's actual Spark DAG execution metrics, cluster hardware config, and DataFrame schema.

## Your Capabilities
1. **Bottleneck Detection**: Identify data skew, shuffle spill, small file problems, and broadcast join opportunities.
2. **Root Cause Analysis**: Map symptoms to root causes natively.
3. **Auto-Execution Generation**: Your code will be executing *directly* in the user's live notebook session.
4. **Safety Awareness**: Prevent OOM operations and out-of-core Cartesian joins.

## Behavioral Rules
- **Be precise**: Reference specific stage IDs, skew ratios, and memory constraints.
- **Show the code**: You MUST include ONE highly optimized, runnable PySpark code block.
- **Assignment Rule (CRITICAL)**: The final optimized DataFrame in your code block MUST be assigned to the variable `df_opt`. If you don't use `df_opt = ...`, the auto-fix engine will fail.
- **Assume Context**: The user's input DataFrame is available in your block as the variable `df`. The `spark` session and PySpark `F` (functions) are also available.
- **Explain**: Briefly explain the 'why' before the code block.
- **Be concise**: Avoid filler. Lead with the diagnosis.
- **Never fabricate metrics**: Only reference data from the injected context.
"""


class ReasoningEngine:
    """Gemini-based reasoning engine for PySpark diagnostics."""

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.models = [
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemma-3-27b"
        ]

    # ───────────────────────────────────────────────────────────────────
    # v0.2.0 — Multi-Turn Chat Session
    # ───────────────────────────────────────────────────────────────────

    def start_chat(self, combined_context):
        """Initialize a Gemini Chat Session grounded with execution context."""
        context_injection = self._build_context_injection(combined_context)
        
        last_exception = None
        for model_id in self.models:
            try:
                chat = self.client.chats.create(
                    model=model_id,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                        max_output_tokens=4096,
                    ),
                )
                # Inject the hidden context prompt to ground the LLM in real metrics
                chat.send_message(context_injection)
                self.model_id = model_id  # Save successful model for future reference if needed
                return chat
            except Exception as e:
                # Catch quota exceeded, model not found, etc.
                last_exception = e
                print(f"Warning: Failed to start chat with {model_id} ({e}). Trying fallback...")
                
        raise RuntimeError(f"All fallback models failed. Last error: {last_exception}")

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
        """Single-turn generation call with fallback logic."""
        last_exception = None
        for model_id in self.models:
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                        max_output_tokens=4096,
                    ),
                )
                self.model_id = model_id
                return response.text
            except Exception as e:
                last_exception = e
                print(f"Warning: Failed to generate content with {model_id} ({e}). Trying fallback...")
                
        raise RuntimeError(f"All fallback models failed. Last error: {last_exception}")

    def _build_context_injection(self, combined_context):
        """Build the hidden context injection from a combined context dict."""
        sections = []

        # DataFrame introspection (primary path)
        df_ctx = combined_context.get("dataframe")
        if df_ctx:
            # Schema
            schema = df_ctx.get("schema")
            if isinstance(schema, list):
                schema_str = "\n".join(
                    f"  - {f['name']}: {f['type']} (nullable={f['nullable']})"
                    for f in schema
                )
            else:
                schema_str = str(schema)

            sections.append(f"""## DataFrame Schema
{schema_str}
Columns: {df_ctx.get('num_columns', '?')} | Partitions: {df_ctx.get('num_partitions', '?')}""")

            # Size
            size_mb = df_ctx.get("estimated_size_mb")
            if size_mb is not None:
                sections.append(f"## Estimated Size\n{size_mb:.4f} MB ({df_ctx.get('estimated_size_bytes', '?')} bytes)")

            # Execution plan
            plan = df_ctx.get("execution_plan")
            if plan and plan != "Could not extract execution plan":
                sections.append(f"## Execution Plan (from Catalyst)\n```\n{plan}\n```")

            # Logical plan
            logical = df_ctx.get("logical_plan")
            if logical:
                sections.append(f"## Optimized Logical Plan\n```\n{logical}\n```")

            # Cluster Settings
            conf = df_ctx.get("spark_conf")
            if conf:
                sections.append(f"## Spark Configuration\n"
                                f"Shuffle Partitions: {conf.get('spark.sql.shuffle.partitions')}\n"
                                f"Driver Memory: {conf.get('spark.driver.memory')}\n"
                                f"Executor Memory: {conf.get('spark.executor.memory')} | "
                                f"Cores: {conf.get('spark.executor.cores')}")

        # DAG metrics (legacy path)
        dag = combined_context.get("dag_metrics")
        if dag:
            if isinstance(dag, list) and dag:
                skewed = [m for m in dag if m.get("skew_ratio", 0) > 3.0]
                critical = [m for m in dag if m.get("skew_ratio", 0) > 5.0]
                sections.append(f"""## DAG Stage Metrics
{json.dumps(dag, indent=2)}

Summary: {len(dag)} stages | {len(skewed)} skewed (>3.0) | {len(critical)} critical (>5.0)""")
            else:
                sections.append(f"## DAG Metrics\n{json.dumps(dag, indent=2)}")

        # Statement text
        stmt = combined_context.get("statement_text")
        if stmt:
            sections.append(f"## PySpark Code Executed\n```python\n{stmt}\n```")

        # General mode
        if not sections:
            sections.append("## Context\nNo DataFrame or metrics context available. The user may provide their code inline.")

        body = "\n\n".join(sections)

        return f"""[SYSTEM CONTEXT INJECTION — INVISIBLE TO USER — DO NOT REFERENCE THIS MESSAGE DIRECTLY]

You are now loaded with the execution context for this interactive session.

{body}

## Instructions
- Use the above context to ground ALL your analysis and recommendations.
- When diagnosing issues, reference specific schema columns, partition counts, and plan operators.
- When generating code fixes, produce exact, runnable PySpark code.
- If a DataFrame was provided, analyze its execution plan for bottlenecks (shuffles, sorts, cartesian products, skew).
- Wait for the user's first question before responding.
- Do NOT acknowledge or reference this system injection message.
"""