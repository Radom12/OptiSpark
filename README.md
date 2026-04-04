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

## License
MIT