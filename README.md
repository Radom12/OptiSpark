# OptiSpark ⚡
**The Autonomous Optimization Layer for PySpark**

OptiSpark is an agentic tool that monitors Spark execution, identifies performance bottlenecks (like Data Skew and Spill), and generates safe, optimized PySpark code to fix them—automatically.

## Features
- 🔍 **Hybrid Extraction**: Supports Standard clusters (EventLogs) and Databricks Serverless (System Tables).
- 🧠 **LLM-Powered Reasoning**: Uses Gemini Flash to diagnose root causes and generate exact PySpark fixes.
- 🛡️ **Catalyst Safety Layer**: Programmatically validates AI-generated code against DataFrame metadata to prevent OOMs.
- 🔒 **AST Execution Blockers**: Static code parsing to prevent data-altering calls (for example, `.save()`, `.saveAsTable()`, `.insertInto()`, `.drop()`, and SQL DDL/DML).
- 🔄 **Catalyst Auto-Healing**: Recursively catches missing column issues or `AnalysisException`s natively to self-correct generated pipelines without human input.
- 💬 **Interactive REPL Agent** *(v0.2.0)*: Chat with OptiSpark in your notebook — ask questions, get context-aware fixes.

## Quick Start

### One-Shot Optimization (v0.1.0)
```python
from optispark import OptiSpark

agent = OptiSpark(api_key="your_gemini_key")
agent.optimize(spark=spark, query_id="your_query_id", target_df=your_df)
```

### Interactive Chat Agent (v0.2.0)
```python
from optispark import OptiSpark

agent = OptiSpark(api_key="your_gemini_key")

# Just pass the DataFrame you're having trouble with!
agent.chat(df=my_problematic_df)

# The agent will automatically introspect:
#   - Schema (column names, types)
#   - Catalyst execution plan
#   - Partition count
#   - Estimated size from Catalyst stats
#
# Available commands inside the REPL:
#   /help      — Show available commands
#   /context   — Display DataFrame context (schema, partitions, size)
#   /plan      — Show the Catalyst execution plan
#   /schema    — Show the full DataFrame schema
#   /clear     — Clear the screen
#   exit       — End the session
```

### CLI Usage
```bash
# Set your API key
export GEMINI_API_KEY="your_key"

# One-shot analysis
optispark analyze --log-dir /path/to/spark/logs

# Interactive chat
optispark chat --log-dir /path/to/spark/logs
```

## Installation
```bash
pip install git+https://github.com/Radom12/OptiSpark.git
```

## Architecture
```
src/optispark/
├── agent.py       # Central orchestrator (optimize + chat REPL)
├── reasoning.py   # Gemini-powered diagnosis and chat engine
├── parser.py      # Hybrid metric extraction (EventLogs + System Tables)
├── safety.py      # Catalyst safety validation layer
├── listener.py    # Real-time Spark listener for task metrics
└── cli.py         # CLI entry point
```


## Known Limitations (V1 Architecture)
```
OptiSpark is currently in version 0.2.0. While the PySpark execution and AST safety layers are highly robust, the FastAPI backend has a few architectural limitations designed for rapid prototyping rather than massive horizontal scale:

1. **In-Memory Session State:** The backend stores active LLM chat sessions in a global Python dictionary (`chat_sessions`). This means the API currently requires a single-worker deployment. If scaled horizontally across multiple workers, sessions will not be shared, resulting in `404 Session not found` errors. 
2. **Ephemeral Hosting (Render Free Tier):** If deploying the backend on free-tier services (like Render) that spin down after inactivity, all active chat sessions will be permanently lost during a cold start due to the in-memory state architecture.
3. **Synchronous Endpoints:** The V1 API endpoints are synchronous, which relies on FastAPI's external threadpool. Under extremely high concurrent load, this could lead to thread exhaustion.

**Future Scaling:** V2 of the backend will introduce an external state store (e.g., Redis) to cache LLM context and session history, allowing for fully stateless API workers and seamless horizontal scaling.
```
