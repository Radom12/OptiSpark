# OptiSpark ⚡
**The Autonomous Optimization Layer for PySpark**

OptiSpark is an agentic tool that monitors Spark execution, identifies performance bottlenecks (like Data Skew and Spill), and generates safe, optimized PySpark code to fix them—automatically.

## Features
- 🔍 **Hybrid Extraction**: Supports Standard clusters (EventLogs) and Databricks Serverless (System Tables).
- 🧠 **LLM-Powered Reasoning**: Uses Gemini 3 Flash to diagnose root causes.
- 🛡️ **Catalyst Safety Layer**: Programmatically validates AI-generated code against DataFrame metadata to prevent OOMs.

## Quick Start
```python
from optispark import OptiSpark

agent = OptiSpark(api_key="your_gemini_key")

# Analyze a specific query
agent.optimize(spark=spark, query_id="your_query_id", target_df=your_df)