"""
OptiSpark Backend API Server
Secure proxy between client packages and Google AI models.
Gemma 4 31B is the primary model; Gemini models are used only for complex fallback.
"""

import os
import uuid
import time
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required on the server.")

# Model priority: Gemma 4 31B first (free, fast), Gemini only for fallback
PRIMARY_MODELS = ["gemma-4-31b-it"]
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-flash-latest"]

SESSION_TTL_SECONDS = 3600

# ═══════════════════════════════════════════════════════════════════════════
# System Prompt (server-side only — never shipped to clients)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are **OptiSpark**, an elite PySpark autonomous execution agent.

## Your Identity
- You are an interactive AI agent embedded in the user's PySpark environment (Databricks, EMR, Dataproc, or local Spark).
- You diagnose Apache Spark bottlenecks and generate **executable** PySpark code fixes.
- You have direct access to the user's actual Spark execution plan, cluster config, and DataFrame schema.

## Your Capabilities
1. **Bottleneck Detection**: Identify data skew, shuffle spill, small file problems, and broadcast join opportunities.
2. **Root Cause Analysis**: Map symptoms to root causes natively.
3. **Auto-Execution Generation**: Your code will be executing *directly* in the user's live notebook session.
4. **Safety Awareness**: Prevent OOM operations and out-of-core Cartesian joins.

## Behavioral Rules
- **Analyze the Full Pipeline**: The `df` you receive might be the final output of a complex pipeline. Use its execution plan to find the ROOT CAUSE (e.g. an upstream skewed join), not just the final aggregation.
- **Reconstruct from Root**: If the bottleneck is upstream, DO NOT just re-process the already-aggregated `df`. Instead, completely reconstruct the fixed pipeline from its sources. The `spark` session object is available.
- **Preserve Output**: `optimized_df` MUST output the exact same schema and semantics as the original `df`.
- **Assignment Rule (CRITICAL)**: Your generated code must end by assigning the final, corrected DataFrame to a variable named `optimized_df`. Do not call `.collect()` or `.show()`.
- **Data Safety Rule (CRITICAL)**: You are strictly forbidden from generating any code that alters data states. Do not use `.write`, `.insertInto()`, `.saveAsTable()`, or any SQL DML/DDL.
- **Ambiguity Resolution**: If the input contains an AnalysisException regarding 'ambiguous' columns, you must rewrite the join logic using `.alias()` on the DataFrames and explicitly reference the aliases in the join condition to resolve the ambiguity.
- **Assume Context**: `df` (input), `spark` (SparkSession), and PySpark `F` are available. 
- **Explain**: Briefly explain the 'why' before the code block. Lead with the diagnosis.
- **Never fabricate metrics**: Only reference data from the injected context.
"""

# ═══════════════════════════════════════════════════════════════════════════
# In-Memory Session Store
# ═══════════════════════════════════════════════════════════════════════════

chat_sessions = {}  # {session_id: {"chat": chat_obj, "model": model_id, "last_activity": timestamp}}

def _get_session_timestamp(session: dict) -> float:
    """Return the most recent activity timestamp for a session, falling back to created_at.

    Returns 0 if neither key is present, so the session is treated as expired.
    """
    return session.get("last_activity", session.get("created_at", 0))

def _cleanup_expired_sessions():
    """Remove sessions older than TTL."""
    now = time.time()
    expired = [sid for sid, s in chat_sessions.items() if now - _get_session_timestamp(s) > SESSION_TTL_SECONDS]
    for sid in expired:
        del chat_sessions[sid]

# ═══════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    print(f"🚀 OptiSpark API Server starting...")
    print(f"   Primary model: {PRIMARY_MODELS[0]}")
    print(f"   Fallback models: {FALLBACK_MODELS}")
    yield
    print("🛑 OptiSpark API Server shutting down.")

app = FastAPI(
    title="OptiSpark API",
    description="Secure backend proxy for OptiSpark PySpark optimization agent",
    version="1.0.0",
    lifespan=lifespan,
)

cors_allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if not cors_allowed_origins:
    import warnings
    warnings.warn(
        "CORS_ALLOWED_ORIGINS is not set; all cross-origin browser requests will be blocked. "
        "Set CORS_ALLOWED_ORIGINS to a comma-separated list of trusted origins.",
        stacklevel=1,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=GEMINI_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════════════════

class ChatStartRequest(BaseModel):
    combined_context: dict = Field(..., description="DataFrame context, DAG metrics, etc.")

class ChatStartResponse(BaseModel):
    session_id: str
    model_used: str

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str

class ChatMessageResponse(BaseModel):
    text: str
    session_id: str

class GenerateRequest(BaseModel):
    prompt: str
    use_fallback: bool = Field(False, description="Use Gemini fallback for complex tasks")

class GenerateResponse(BaseModel):
    text: str
    model_used: str

class HealthResponse(BaseModel):
    status: str
    primary_model: str

# ═══════════════════════════════════════════════════════════════════════════
# Context Injection Builder (server-side only)
# ═══════════════════════════════════════════════════════════════════════════

def _build_context_injection(combined_context: dict) -> str:
    """Build the hidden context injection from a combined context dict."""
    sections = []

    df_ctx = combined_context.get("dataframe")
    if df_ctx:
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

        size_mb = df_ctx.get("estimated_size_mb")
        if size_mb is not None:
            sections.append(f"## Estimated Size\n{size_mb:.4f} MB ({df_ctx.get('estimated_size_bytes', '?')} bytes)")

        plan = df_ctx.get("execution_plan")
        if plan and plan != "Could not extract execution plan":
            sections.append(f"## Execution Plan (from Catalyst)\n```\n{plan}\n```")

        logical = df_ctx.get("logical_plan")
        if logical:
            sections.append(f"## Optimized Logical Plan\n```\n{logical}\n```")

        parsed = df_ctx.get("parsed_logical_plan")
        if parsed:
            sections.append(f"## Parsed (Original) Logical Plan\n```\n{parsed}\n```")

        conf = df_ctx.get("spark_conf")
        if conf:
            sections.append(f"## Spark Configuration\n"
                            f"Shuffle Partitions: {conf.get('spark.sql.shuffle.partitions')}\n"
                            f"Driver Memory: {conf.get('spark.driver.memory')}\n"
                            f"Executor Memory: {conf.get('spark.executor.memory')} | "
                            f"Cores: {conf.get('spark.executor.cores')}")

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

    stmt = combined_context.get("statement_text")
    if stmt:
        sections.append(f"## PySpark Code Executed\n```python\n{stmt}\n```")

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

# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", primary_model=PRIMARY_MODELS[0])


@app.post("/api/v1/chat/start", response_model=ChatStartResponse)
def start_chat(req: ChatStartRequest):
    """Create a new chat session grounded with execution context."""
    _cleanup_expired_sessions()

    context_injection = _build_context_injection(req.combined_context)
    all_models = PRIMARY_MODELS + FALLBACK_MODELS
    last_exception = None

    for model_id in all_models:
        try:
            chat = client.chats.create(
                model=model_id,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
            )
            chat.send_message(context_injection)

            session_id = str(uuid.uuid4())
            chat_sessions[session_id] = {
                "chat": chat,
                "model": model_id,
                "last_activity": time.time(),
            }
            return ChatStartResponse(session_id=session_id, model_used=model_id)
        except Exception as e:
            last_exception = e
            continue

    raise HTTPException(status_code=503, detail=f"All models failed: {last_exception}")


@app.post("/api/v1/chat/message", response_model=ChatMessageResponse)
def send_message(req: ChatMessageRequest):
    """Send a message to an existing chat session."""
    session = chat_sessions.get(req.session_id)
    if not session or time.time() - _get_session_timestamp(session) > SESSION_TTL_SECONDS:
        if req.session_id in chat_sessions:
            del chat_sessions[req.session_id]
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    try:
        response = session["chat"].send_message(req.message)
        session["last_activity"] = time.time()  # Refresh TTL
        return ChatMessageResponse(text=response.text, session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


@app.post("/api/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Single-turn generation for diagnose/fix operations."""
    models = (FALLBACK_MODELS + PRIMARY_MODELS) if req.use_fallback else (PRIMARY_MODELS + FALLBACK_MODELS)
    last_exception = None

    for model_id in models:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=req.prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
            )
            return GenerateResponse(text=response.text, model_used=model_id)
        except Exception as e:
            last_exception = e
            continue

    raise HTTPException(status_code=503, detail=f"All models failed: {last_exception}")
