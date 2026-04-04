# OptiSpark: An Autonomous Copilot for PySpark Optimization

OptiSpark is an agentic tool that monitors Spark execution, identifies performance bottlenecks (such as Data Skew and Spill), and generates safe, optimized PySpark code to fix them—automatically. 

Built with a focus on robust execution and cluster safety, OptiSpark serves as a highly capable, developer-focused copilot for rapidly iterating on complex data engineering pipelines.

## Core Engineering Features

OptiSpark isn't just an LLM wrapper; it tightly integrates with PySpark's internal execution engine to ensure safe, context-aware optimizations.

- 🔒 **Abstract Syntax Tree (AST) Safety (ReadOnlyValidator)** 
  Before any LLM-generated code touches your live SparkSession, our `ast.NodeVisitor` parses the syntax tree to build a strictly read-only execution sandbox. It intercepts and blocks destructive DataFrame operations (e.g., `.write`, `.save()`, `.drop()`) and malicious SQL DDL/DML tokens natively.
- ⚙️ **Deep Catalyst Optimizer Integration** 
  OptiSpark programmatically taps into the Spark Catalyst Optimizer to extract execution metadata (EventLogs and Databricks System Tables), logical/physical plans, and DAG execution metrics. It catches and self-heals from `AnalysisException` errors (like ambiguous column joins) entirely natively.
- 🛡️ **Pre-Execution Cost Estimation** 
  To prevent cluster Out-of-Memory (OOM) errors during sandbox validation, OptiSpark evaluates `.explain(mode="cost")` to preemptively intercept exponentially huge cartesian explosions caused by unchecked `F.explode()` operations.
- 🧠 **LLM-Powered Reasoning** 
  Uses Gemini Flash to diagnose root causes based on Catalyst stats and generate exact PySpark fixes.

## Quick Start

### One-Shot Optimization
```python
from optispark import OptiSpark

agent = OptiSpark()
agent.optimize(spark=spark, query_id="your_query_id", target_df=your_df)
```

### Interactive Chat Agent
```python
from optispark import OptiSpark

agent = OptiSpark()

# Just pass the DataFrame you're having trouble with!
agent.chat(df=my_problematic_df)

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
# One-shot analysis
optispark analyze --log-dir /path/to/spark/logs

# Interactive chat
optispark chat --log-dir /path/to/spark/logs
```

## Installation
```bash
pip install git+https://github.com/Radom12/OptiSpark.git
```

## Architecture Layout
```text
src/optispark/
├── agent.py       # Central orchestrator (optimize + chat REPL)
├── reasoning.py   # Gemini-powered diagnosis and chat engine
├── parser.py      # Hybrid metric extraction (EventLogs + System Tables)
├── safety.py      # AST parsing, secure sandbox, and Catalyst safety layer
├── listener.py    # Real-time Spark listener for task metrics
└── cli.py         # CLI entry point
```

## Known Limitations (V1 Architecture)

OptiSpark's core PySpark extraction and AST safety layers are highly robust. However, the current V1 FastAPI backend was engineered for functional prototyping and has the following architectural bottlenecks:

- **In-Memory Session State**: The backend stores active LLM chat sessions in a global Python dictionary (`chat_sessions`). This restricts deployment to a single API worker; attempting to scale horizontally across multiple workers will result in `404 Session not found` errors as state is not shared.
- **Ephemeral Hosting Constraints**: Because of the in-memory state dependency, deploying on free-tier services (such as Render) that spin down after short periods of inactivity will cause all active user chat sessions to be permanently lost during a cold start.
- **Synchronous Endpoints**: The V1 API endpoints process requests synchronously using FastAPI's default external threadpool for blocking operations. Under extreme concurrent load, this architecture risks thread exhaustion and timeouts.

## Roadmap

- **Stateless Backend (Redis)**: Migrate the in-memory chat session state to an external Redis cache to support concurrent users and stateless horizontal API scaling.
- **PySpark 4.0 Compatibility**: Test and upgrade internal APIs ahead of Spark 4.0 release.
- **Enhanced Telemetry Extraction**: Integrate directly with Databricks SQL warehouses for near-realtime metric fetching.

## License
MIT
