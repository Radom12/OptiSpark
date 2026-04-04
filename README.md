# OptiSpark: An Autonomous Copilot for PySpark Optimization

OptiSpark is an agentic tool that monitors Spark execution, identifies performance bottlenecks (such as Data Skew, Shuffle Spill, and missing Broadcast Joins), and generates safe, optimized PySpark code to fix them—automatically.

Built with a focus on robust execution and cluster safety, OptiSpark serves as a highly capable, developer-focused copilot for rapidly debugging and fixing complex PySpark pipelines.

## Core Engineering

### AST-Based Read-Only Sandbox (`ReadOnlyValidator`)

All LLM-generated code is parsed through a `ReadOnlyValidator` — a Python `ast.NodeVisitor` subclass — before it reaches `exec()`. The validator walks the full syntax tree and blocks:

- **Destructive attribute access**: `.write` (intercepted at `visit_Attribute`)
- **Destructive method calls**: `.save()`, `.saveAsTable()`, `.insertInto()`, `.drop()`, `.delete()`, `.truncate()` (intercepted at `visit_Call`)
- **Destructive SQL**: `DROP`, `DELETE`, `TRUNCATE`, `INSERT`, `UPDATE`, `CREATE` tokens inside `spark.sql()` string arguments

If any violation is detected, a `ValueError` is raised and the code is never executed.

### Catalyst Optimizer Integration

OptiSpark programmatically taps into the Spark Catalyst engine via Py4J to extract execution metadata without relying on the Spark UI:

- **Execution Plan**: `df._jdf.queryExecution().toString()` (full optimized + physical plan)
- **Logical Plan**: `df._jdf.queryExecution().optimizedPlan().toString()`
- **Estimated Size**: `df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()`
- **Partition Count**: `df.rdd.getNumPartitions()`
- **Cluster Config**: `spark.conf.get(...)` for `shuffle.partitions`, `driver.memory`, `executor.memory`, `executor.cores`

### Pre-Execution Cost Estimation

Before sandbox execution, `validate_safety()` checks whether the generated code contains high-risk operations (e.g., `F.explode`). If it does, it queries the Catalyst `optimizedPlan().stats().sizeInBytes()` to estimate the DataFrame size and blocks execution if it exceeds a configurable threshold (default: 50 MB), preventing OOM errors.

### Self-Healing Execution Loop

When sandbox execution throws a PySpark `AnalysisException` (e.g., ambiguous column references from multi-table joins), the agent automatically:

1. Catches the exception
2. Feeds the full error traceback back to the LLM with targeted repair instructions (e.g., "use `.alias()` on DataFrames and explicitly reference aliases in the join condition")
3. Extracts the corrected code block from the LLM response
4. Retries execution (up to `max_retries=3`)

This loop runs invisibly — the end user sees only the final working result.

## Quick Start

### Interactive Chat Agent (Primary Usage)

```python
from optispark import OptiSpark

agent = OptiSpark()

# Pass the DataFrame you're having trouble with.
# The agent introspects schema, execution plan, partitions, and Catalyst stats.
optimized_df = agent.chat(df=my_problematic_df)

# Pass upstream source DataFrames as kwargs for full pipeline reconstruction:
optimized_df = agent.chat(
    df=final_df,
    df_transactions=df_transactions,
    df_users=df_users,
    df_logs=df_logs
)
```

**REPL Commands** (available inside the interactive session):

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/context` | Display DataFrame context (schema, partitions, size) |
| `/plan` | Show the Catalyst execution plan |
| `/schema` | Show the full DataFrame schema |
| `/benchmark` | Benchmark the last generated fix against the original |
| `/clear` | Clear the screen |
| `exit` | End session and return the optimized DataFrame |

### One-Shot Optimization (Legacy)

```python
from optispark import OptiSpark

agent = OptiSpark(log_dir="/path/to/spark/event_logs")
agent.optimize(target_df=your_df)
```

The one-shot mode extracts metrics from Spark event logs, sends them to the LLM for diagnosis, generates a fix, validates it through the Catalyst safety layer, and prints the suggested code. It does **not** auto-execute.

### CLI

```bash
# One-shot analysis from event logs
optispark analyze --log-dir /path/to/spark/logs

# Interactive chat from event logs
optispark chat --log-dir /path/to/spark/logs

# Point to a custom backend
optispark chat --log-dir /path/to/spark/logs --server-url http://localhost:8000
```

### Constructor Parameters

```python
OptiSpark(log_dir=None, server_url=None)
```

| Parameter | Type | Description |
|---|---|---|
| `log_dir` | `str` or `None` | Path to local Spark event logs directory (for standard clusters). |
| `server_url` | `str` or `None` | Custom backend API URL. Falls back to `OPTISPARK_SERVER_URL` env var, then to the default production endpoint. |

> **Note**: No API key is required on the client side. The OptiSpark backend manages all LLM credentials server-side.

## Installation

```bash
pip install optispark
```

**Requirements**: Python ≥ 3.9, PySpark ≥ 3.0

## Architecture

```text
src/optispark/
├── agent.py       # Central orchestrator: OptiSpark class, REPL loop, sandbox execution
├── reasoning.py   # HTTP client that proxies to the OptiSpark backend API
├── parser.py      # Hybrid metric extraction (EventLogs + Databricks System Tables)
├── safety.py      # ReadOnlyValidator (AST), secure_exec(), Catalyst cost estimation
├── benchmark.py   # Sampled dry-run benchmark engine (original vs. optimized)
├── listener.py    # Real-time Spark listener for task-level metrics
└── cli.py         # CLI entry point (argparse)

server/
└── main.py        # FastAPI backend: Gemini/Gemma proxy, session management, system prompt
```

## Known Limitations (V1 Architecture)

The PySpark extraction engine and AST safety layers are robust. However, the V1 FastAPI backend has architectural constraints designed for rapid prototyping rather than horizontal scale:

- **In-Memory Session State**: The backend stores active LLM chat sessions in a global Python dictionary (`chat_sessions`). This restricts deployment to a single API worker. Scaling horizontally across multiple workers will result in `404 Session not found` errors because state is not shared between processes.
- **Ephemeral Hosting Constraints**: Because of the in-memory state dependency, deploying on free-tier services (such as Render) that spin down after inactivity will cause all active chat sessions to be permanently lost during a cold start.
- **Synchronous Endpoints**: The V1 API endpoints process requests synchronously using FastAPI's default external threadpool. Under extreme concurrent load, this could lead to thread exhaustion and request timeouts.

## Roadmap

- **Stateless Backend (Redis)**: Migrate the in-memory chat session state to an external Redis cache to support concurrent users and stateless horizontal API scaling.
- **PySpark 4.0 Compatibility**: Test and upgrade internal Py4J APIs ahead of the Spark 4.0 release.
- **Enhanced Telemetry**: Integrate directly with Databricks SQL warehouses for near-realtime metric extraction.

## License

MIT
