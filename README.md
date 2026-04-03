# OptiSpark: Enterprise-Grade PySpark Optimization Agent

**OptiSpark** is an autonomous, agentic LLM data quality and remediation engine for PySpark. Designed specifically for enterprise Databricks and AWS environments, it acts as an interactive "copilot" that dynamically diagnoses execution bottlenecks, handles Catalyst analyzer failures, and securely generates optimized PySpark code.

Unlike standard code generators, OptiSpark utilizes a strictly sandboxed **Abstract Syntax Tree (AST) execution environment** and leverages the **Spark Catalyst Optimizer** to predict memory costs—ensuring zero destructive operations and preventing out-of-memory (OOM) errors before they hit your cluster.

## Core Features

*   **Enterprise Security Sandbox:** An integrated AST parser statically intercepts and blocks all destructive write/drop operations (`df.write`, `DROP TABLE`, etc.), strictly enforcing a read-only execution environment.
    
*   **Agentic Diagnostics:** Automatically extracts and analyzes Catalyst logical/physical plans, DAG execution metrics, and schemas to diagnose root causes (e.g., data skew, shuffle spill, exploding joins).
    
*   **Early Analyzer Resolution:** Catches `AnalysisException` errors (such as Ambiguous Joins) at the Catalyst Analyzer phase and autonomously rewrites column aliasing to resolve them.
    
*   **Pre-Execution Cost Estimation:** Uses `df.explain(mode="cost")` to evaluate the optimized plan's estimated size in bytes against hardcoded safety thresholds, preventing catastrophic memory explosions.
    
*   **Decoupled Architecture:** Client logic runs locally in your PySpark notebook, while LLM reasoning and system prompts are securely proxied through a FastAPI backend.
    

## System Architecture

OptiSpark uses a Client-Server model to protect API keys and prevent prompt injection:

1.  **Backend Server (FastAPI):** Houses the LLM integration (Gemma 4 / Gemini models), the core reasoning engine, and the proprietary system prompts.
    
2.  **PySpark Client:** Injected into your Databricks/Jupyter environment. Extracts query plans, runs the AST validator, and executes the authorized code safely.
    

## Setup & Installation

### 1\. Backend Server Deployment

The backend proxy must be running to handle reasoning requests. You can run this locally or deploy it to a cloud provider (e.g., AWS ECS, Render).

    # Clone the repository
    git clone [https://github.com/yourusername/optispark.git](https://github.com/yourusername/optispark.git)
    cd optispark
    
    # Navigate to the server directory
    cd server
    
    # Install backend dependencies
    pip install -r requirements.txt
    
    # Set your LLM API Key (e.g., Google Gemini)
    export GEMINI_API_KEY="your_api_key_here"
    
    # Run the FastAPI server (defaults to http://localhost:8000)
    uvicorn main:app --reload
    

### 2\. Client Installation

Install the OptiSpark client package in your PySpark environment (Databricks, EMR, or local Jupyter).

    # From the root of the repository
    pip install -e .
    

Configure the client to point to your backend server:

    import os
    os.environ["OPTISPARK_SERVER_URL"] = "http://localhost:8000" # Or your deployed URL
    

## Usage Guide

OptiSpark is designed to be imported directly into your active PySpark notebook.

### Basic Optimization

Pass an unoptimized DataFrame to the agent. OptiSpark will analyze the execution plan, suggest an optimization (like salting a skewed join), validate the code via AST, and return the optimized DataFrame.

    from optispark.agent import OptiSparkAgent
    
    # Your slow or failing dataframe
    df_skewed = transactions.join(users, on="user_id", how="left")
    
    # Initialize the agent
    agent = OptiSparkAgent()
    
    # The agent returns a safely executed, optimized DataFrame
    df_opt = agent.optimize(df_skewed)
    
    # Proceed with your pipeline
    df_opt.write.saveAsTable("trusted_transactions")
    

### Handling Ambiguous Joins (Analyzer Exceptions)

If your initial code fails during the Catalyst evaluation phase, OptiSpark catches the trace and fixes it autonomously.

    # Fails with AnalysisException: Reference 'id' is ambiguous
    df_ambiguous = table_a.join(table_b, table_a.id == table_b.id).select("id")
    
    df_opt = agent.optimize(df_ambiguous) 
    # Agent automatically aliases tables and outputs the corrected logical plan
    

## Security & Guardrails

OptiSpark was built with strict production guardrails:

1.  **No `exec()` vulnerability:** All LLM-generated code is routed through `optispark.safety.ReadOnlyValidator`. If the LLM attempts to generate a `.drop()`, `.write()`, or destructive SQL command, the AST parser raises a `ValueError` before execution.
    
2.  **Output Enforcement:** The agent is constrained to assign its final output to `df_opt`. It is prevented from calling compute-heavy actions like `.collect()` or `.show()` internally.
    
3.  **Max Size Thresholds:** You can configure `max_safe_size_mb` in the safety module. The agent will refuse to return code that Catalyst estimates will exceed this bound.
    

## Roadmap

*   \[ \] **AWS Native Deployment:** Terraform scripts for deploying the FastAPI backend to AWS ECS/API Gateway.
    
*   \[ \] **Databricks Unity Catalog:** Integrate with Unity Catalog to analyze table metadata natively without triggering cluster compute.
    
*   \[ \] **Artifact Generation:** Option to output the generated fixes as a `.py` file artifact or GitHub Pull Request rather than executing dynamically.
    

## Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE "null") file for details.
