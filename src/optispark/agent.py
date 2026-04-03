"""
OptiSpark Agent — The central orchestrator for the autonomous PySpark optimization agent.
Supports both one-shot optimization (v0.1.0) and interactive REPL chat (v0.2.0).
"""

import os
import json
import textwrap
from datetime import datetime

from .reasoning import ReasoningEngine
from .parser import extract_features_from_logs, extract_features_from_system_tables
from .safety import validate_safety


# ─── ANSI Color Palette ──────────────────────────────────────────────────────
class _Colors:
    RESET     = "\033[0m"
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    ITALIC    = "\033[3m"
    UNDERLINE = "\033[4m"

    # Premium dark-mode palette
    CYAN      = "\033[38;5;81m"
    ORANGE    = "\033[38;5;214m"
    GREEN     = "\033[38;5;114m"
    RED       = "\033[38;5;203m"
    MAGENTA   = "\033[38;5;176m"
    YELLOW    = "\033[38;5;228m"
    BLUE      = "\033[38;5;111m"
    GRAY      = "\033[38;5;245m"
    WHITE     = "\033[38;5;255m"

    # Background accents
    BG_DARK   = "\033[48;5;235m"
    BG_BLUE   = "\033[48;5;23m"
    BG_GREEN  = "\033[48;5;22m"
    BG_RED    = "\033[48;5;52m"

C = _Colors


class OptiSpark:
    """
    The OptiSpark Agent.

    Usage:
        agent = OptiSpark()

        # One-shot optimization (v0.1.0)
        agent.optimize(spark=spark, query_id="...", target_df=df)

        # Interactive REPL chat (v0.2.0)
        agent.chat(df=my_problematic_df)
    """

    def __init__(self, log_dir=None, server_url=None):
        """Initialize the OptiSpark agent.

        Args:
            log_dir: Path to local Spark event logs (for Standard clusters).
            server_url: Optional custom backend URL. Uses default if not provided.
        """
        self.log_dir = log_dir
        self.engine = ReasoningEngine(server_url=server_url)

    # ═══════════════════════════════════════════════════════════════════════
    # v0.1.0 — One-Shot Optimization
    # ═══════════════════════════════════════════════════════════════════════

    def optimize(self, spark=None, query_id=None, target_df=None):
        """Run a single-shot analysis and optimization cycle."""
        print(f"\n{C.CYAN}{C.BOLD}⚡ OptiSpark One-Shot Analysis{C.RESET}")
        print(f"{C.DIM}{'─' * 50}{C.RESET}\n")

        # 1. Extract
        print(f"  {C.BLUE}[1/4]{C.RESET} Extracting execution metrics...")
        features = self._extract_context(spark, query_id)
        if not features:
            print(f"  {C.YELLOW}⚠  No metrics found. Aborting.{C.RESET}")
            return

        # 2. Diagnose
        print(f"  {C.BLUE}[2/4]{C.RESET} Diagnosing bottlenecks via Gemini...")
        diagnosis = self.engine.diagnose(features)
        print(f"  {C.GREEN}✔  Diagnosis complete.{C.RESET}\n")
        print(f"  {C.WHITE}{diagnosis}{C.RESET}\n")

        # 3. Generate fix
        print(f"  {C.BLUE}[3/4]{C.RESET} Generating optimized PySpark code...")
        fix = self.engine.generate_fix(features)

        # 4. Safety check
        print(f"  {C.BLUE}[4/4]{C.RESET} Running Catalyst safety validation...")
        is_safe, safety_msg = validate_safety(fix, target_df)
        if is_safe:
            print(f"  {C.GREEN}✔  {safety_msg}{C.RESET}\n")
            _print_code_block(fix, title="Suggested Fix")
        else:
            print(f"  {C.RED}✖  {safety_msg}{C.RESET}")
            print(f"  {C.YELLOW}⚠  Fix blocked by safety layer.{C.RESET}")

    # ═══════════════════════════════════════════════════════════════════════
    # v0.2.0 — Interactive REPL Chat
    # ═══════════════════════════════════════════════════════════════════════

    def chat(self, df=None, spark=None, query_id=None):
        """Launch the interactive OptiSpark REPL agent.

        Pass the DataFrame you're having trouble with — the agent will
        introspect it (schema, execution plan, partitions, Catalyst stats)
        and use that context to diagnose and fix your Spark performance issues.

        Args:
            df: The PySpark DataFrame you want the agent to analyze and fix.
            spark: (Optional) Active SparkSession — inferred from df if not provided.
            query_id: (Optional, legacy) Databricks query ID for system table lookup.
        """
        _print_banner()

        # ── 1. Context Gathering ──────────────────────────────────────────
        print(f"  {C.BLUE}◆{C.RESET} Introspecting DataFrame...", end="")

        df_context = None
        features = None
        code_context = None

        if df is not None:
            # Primary path: extract everything from the live DataFrame
            df_context = self._introspect_dataframe(df)
            print(f" {C.GREEN}✔{C.RESET}")
            print(f"  {C.GRAY}├─ Columns: {df_context['num_columns']}{C.RESET}")
            print(f"  {C.GRAY}├─ Partitions: {df_context['num_partitions']}{C.RESET}")
            size_mb = df_context.get('estimated_size_mb')
            if size_mb is not None:
                print(f"  {C.GRAY}├─ Estimated size: {size_mb:.2f} MB{C.RESET}")
            print(f"  {C.GRAY}└─ Execution plan: Captured{C.RESET}")
        elif spark and query_id:
            # Legacy path: system table lookup
            features = self._extract_context(spark, query_id)
            code_context = self._fetch_statement_text(spark, query_id)
            if features:
                print(f" {C.GREEN}✔{C.RESET}")
                skewed = [f for f in features if f.get("skew_ratio", 0) > 3.0]
                print(f"  {C.GRAY}├─ Stages: {len(features)}{C.RESET}")
                if skewed:
                    print(f"  {C.ORANGE}├─ ⚠ Skewed: {len(skewed)}{C.RESET}")
                print(f"  {C.GRAY}└─ Code: {'Available' if code_context else 'N/A'}{C.RESET}")
            else:
                print(f" {C.YELLOW}⚠{C.RESET}")
                print(f"  {C.YELLOW}└─ No metrics found.{C.RESET}")
        elif self.log_dir:
            features = extract_features_from_logs(self.log_dir)
            if features:
                print(f" {C.GREEN}✔{C.RESET} (from event logs)")
            else:
                print(f" {C.YELLOW}⚠{C.RESET}")
        else:
            print(f" {C.YELLOW}⚠{C.RESET}")
            print(f"  {C.YELLOW}└─ No DataFrame or context provided — chatting in general mode.{C.RESET}")

        # Build the combined context for the LLM
        combined_context = self._build_combined_context(df_context, features, code_context)

        # ── 2. Initialize Chat Session ────────────────────────────────────
        print(f"\n  {C.BLUE}◆{C.RESET} Initializing Gemini chat session...", end="")
        try:
            chat_session = self.engine.start_chat(combined_context)
            print(f" {C.GREEN}✔{C.RESET}")
        except Exception as e:
            print(f" {C.RED}✖{C.RESET}")
            print(f"\n  {C.RED}Error: {str(e)}{C.RESET}")
            return

        # ── 3. Session State ──────────────────────────────────────────────
        session_state = {
            "message_count": 0,
            "start_time": datetime.now(),
            "df_context": df_context,
            "features": features,
            "code_context": code_context,
        }

        _print_ready_prompt()

        # ── 4. REPL Loop ─────────────────────────────────────────────────
        while True:
            try:
                user_input = input(f"\n  {C.CYAN}{C.BOLD}❯{C.RESET} ")

                # Handle empty input
                if not user_input.strip():
                    continue

                # Handle commands
                cmd = user_input.strip().lower()
                if cmd in ("exit", "quit", "q", "/exit", "/quit"):
                    _print_goodbye(session_state)
                    break
                elif cmd in ("/help", "/h"):
                    _print_help()
                    continue
                elif cmd in ("/metrics", "/m", "/context"):
                    _print_context(session_state)
                    continue
                elif cmd in ("/plan", "/p"):
                    _print_plan(session_state)
                    continue
                elif cmd in ("/schema", "/s"):
                    _print_schema(session_state)
                    continue
                elif cmd in ("/benchmark", "/b"):
                    if "last_code" not in session_state:
                        print(f"  {C.YELLOW}⚠  No AI code generated yet. Ask OptiSpark for a fix first!{C.RESET}")
                        continue
                    if df is None:
                        print(f"  {C.YELLOW}⚠  No original DataFrame found for benchmarking.{C.RESET}")
                        continue
                        
                    from optispark.benchmark import run_benchmark
                    print(f"\n  {C.BLUE}{C.BOLD}🚀 Launching Benchmark Engine{C.RESET}")
                    results = run_benchmark(df, session_state["last_code"])
                    _print_benchmark_results(results)
                    continue
                elif cmd in ("/clear",):
                    _clear_screen()
                    _print_banner()
                    _print_ready_prompt()
                    continue

                # Send to Gemini
                session_state["message_count"] += 1
                _print_thinking()

                response = chat_session.send_message(user_input)
                _print_response(response.text, session_state["message_count"])

                # Extract python blocks globally for benchmarking
                import re
                blocks = re.findall(r"```python\n(.*?)```", response.text, re.DOTALL)
                if blocks:
                    # Save the last generated block for /benchmark runs
                    session_state["last_code"] = blocks[-1]

                # Auto-Execute Sandbox (v0.3.0)
                if df is not None:
                    for block in blocks:
                        if "df_opt" in block:
                            print(f"\n  {C.YELLOW}{C.BOLD}⚡ OptiSpark generated an executable fix.{C.RESET}")
                            confirm = input(f"  {C.BLUE}❯ Apply this code to your DataFrame? [y/N]: {C.RESET}").strip().lower()
                            if confirm == 'y':
                                print(f"  {C.BLUE}◆{C.RESET} Executing securely in background...", end="")
                                try:
                                    import pyspark.sql.functions as F
                                    from pyspark.sql import Window
                                    
                                    # Provide secure local environment with standard PySpark aliases
                                    local_env = {"df": df, "spark": df.sparkSession, "F": F, "Window": Window}
                                    exec(block.strip(), {}, local_env)
                                    
                                    if "df_opt" in local_env:
                                        session_state["df_opt"] = local_env["df_opt"]
                                        df = session_state["df_opt"]  # Update REPL reference
                                        print(f" {C.GREEN}✔ Success!{C.RESET}")
                                        print(f"  {C.GRAY}├─ The optimized DataFrame is now active in this chat session.{C.RESET}")
                                        print(f"  {C.GRAY}└─ It will be returned when you type 'exit'.{C.RESET}")
                                    else:
                                        print(f" {C.RED}✖ Error: The code executed but did not assign 'df_opt'.{C.RESET}")
                                except Exception as ex:
                                    print(f" {C.RED}✖ Execution Failed{C.RESET}")
                                    print(f"  {C.RED}Error: {str(ex)}{C.RESET}")
                                    print(f"\n  {C.GRAY}Feeding the stack trace back to the agent...{C.RESET}")
                                    
                                    # Self-Healing loop
                                    error_prompt = f"The code you provided failed to execute with this error:\n```\n{str(ex)}\n```\nPlease fix the code and output a new python block assigning the result to `df_opt`."
                                    session_state["message_count"] += 1
                                    _print_thinking()
                                    correction = chat_session.send_message(error_prompt)
                                    _print_response(correction.text, session_state["message_count"])
                            break  # Only ask for the first valid block

            except KeyboardInterrupt:
                print()
                _print_goodbye(session_state)
                break
            except Exception as e:
                _print_error(str(e))

        # Return the optimized DataFrame, or fallback to the original
        return session_state.get("df_opt", df)

    # ═══════════════════════════════════════════════════════════════════════
    # Private Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _introspect_dataframe(self, df):
        """Extract rich metadata from a live PySpark DataFrame using Catalyst."""
        context = {}

        # Schema
        try:
            schema_fields = []
            for field in df.schema.fields:
                schema_fields.append({
                    "name": field.name,
                    "type": str(field.dataType),
                    "nullable": field.nullable,
                })
            context["schema"] = schema_fields
            context["num_columns"] = len(schema_fields)
        except Exception:
            context["schema"] = "Could not extract schema"
            context["num_columns"] = "?"

        # Execution plan (the explain output) — truncate to avoid massive payloads
        MAX_PLAN_LEN = 4000
        try:
            plan = df._jdf.queryExecution().toString()
            context["execution_plan"] = plan[:MAX_PLAN_LEN] if len(plan) > MAX_PLAN_LEN else plan
        except Exception:
            try:
                import io
                from contextlib import redirect_stdout
                buf = io.StringIO()
                with redirect_stdout(buf):
                    df.explain(mode="extended")
                plan = buf.getvalue()
                context["execution_plan"] = plan[:MAX_PLAN_LEN] if len(plan) > MAX_PLAN_LEN else plan
            except Exception:
                context["execution_plan"] = "Could not extract execution plan"

        # Partition count (can trigger lazy evaluation — use thread timeout)
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: df.rdd.getNumPartitions())
                context["num_partitions"] = future.result(timeout=10)
        except Exception:
            context["num_partitions"] = "?"

        # Catalyst estimated size (from the optimized logical plan)
        try:
            size_bytes = df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()
            context["estimated_size_bytes"] = int(size_bytes)
            context["estimated_size_mb"] = round(int(size_bytes) / (1024 * 1024), 4)
        except Exception:
            context["estimated_size_bytes"] = None
            context["estimated_size_mb"] = None

        # Logical plan string (more readable)
        try:
            context["logical_plan"] = df._jdf.queryExecution().optimizedPlan().toString()
        except Exception:
            context["logical_plan"] = None

        # Spark cluster configuration and capacity
        try:
            spark = df.sparkSession
            context["spark_conf"] = {
                "spark.sql.shuffle.partitions": spark.conf.get("spark.sql.shuffle.partitions", "default"),
                "spark.driver.memory": spark.conf.get("spark.driver.memory", "default"),
                "spark.executor.memory": spark.conf.get("spark.executor.memory", "default"),
                "spark.executor.cores": spark.conf.get("spark.executor.cores", "default"),
            }
        except Exception:
            context["spark_conf"] = None

        return context

    def _extract_context(self, spark=None, query_id=None):
        """Extract DAG features from the best available source."""
        if spark and query_id:
            return extract_features_from_system_tables(spark, query_id)
        elif self.log_dir:
            return extract_features_from_logs(self.log_dir)
        return None

    def _fetch_statement_text(self, spark=None, query_id=None):
        """Fetch the PySpark statement_text from Databricks system tables."""
        if not spark or not query_id:
            return None
        try:
            row = spark.sql(
                f"SELECT statement_text FROM system.query.history "
                f"WHERE query_id = '{query_id}'"
            ).collect()[0]
            return row["statement_text"]
        except Exception:
            return None

    def _build_combined_context(self, df_context=None, features=None, code_context=None):
        """Merge all available context into a single dict for the LLM."""
        context = {}
        if df_context:
            context["dataframe"] = df_context
        if features:
            context["dag_metrics"] = features
        if code_context:
            context["statement_text"] = code_context
        if not context:
            context["status"] = "No context available — general mode"
        return context


# ═══════════════════════════════════════════════════════════════════════════
# UI Rendering Functions — Premium Terminal UX
# ═══════════════════════════════════════════════════════════════════════════

def _print_banner():
    """Print the OptiSpark welcome banner."""
    banner = f"""
{C.CYAN}{C.BOLD}
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ⚡  O P T I S P A R K   v0.2.0                        ║
    ║        Interactive PySpark Performance Agent               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝{C.RESET}
{C.GRAY}    Powered by Gemini · Catalyst Safety Layer · DAG Analytics{C.RESET}
"""
    print(banner)


def _print_ready_prompt():
    """Print the ready message after initialization."""
    print(f"""
{C.DIM}  {'─' * 55}{C.RESET}
{C.GREEN}{C.BOLD}  ✦ Agent Online{C.RESET} — Ask about your Spark performance.
{C.GRAY}    Type {C.WHITE}/help{C.GRAY} for commands · {C.WHITE}exit{C.GRAY} to quit{C.RESET}
{C.DIM}  {'─' * 55}{C.RESET}""")


def _print_help():
    """Print the help menu."""
    print(f"""
{C.CYAN}{C.BOLD}  ┌─ Commands ──────────────────────────────────────────┐{C.RESET}
{C.WHITE}  │  /help, /h       {C.GRAY}Show this help menu               {C.WHITE}│{C.RESET}
{C.WHITE}  │  /context, /m    {C.GRAY}Show DataFrame & session context  {C.WHITE}│{C.RESET}
{C.WHITE}  │  /plan, /p       {C.GRAY}Show the Catalyst execution plan  {C.WHITE}│{C.RESET}
{C.WHITE}  │  /schema, /s     {C.GRAY}Show the DataFrame schema         {C.WHITE}│{C.RESET}
{C.WHITE}  │  /clear          {C.GRAY}Clear the screen                  {C.WHITE}│{C.RESET}
{C.WHITE}  │  exit, quit, q   {C.GRAY}End the session                   {C.WHITE}│{C.RESET}
{C.CYAN}  └────────────────────────────────────────────────────┘{C.RESET}

{C.GRAY}  💡 Try asking:{C.RESET}
{C.ITALIC}{C.MAGENTA}     "What bottlenecks do you see in my DataFrame?"
     "Fix the data skew in my join"
     "Optimize this query to reduce shuffle"{C.RESET}
""")


def _print_context(session_state):
    """Print all available context (DataFrame + DAG metrics)."""
    df_ctx = session_state.get("df_context")
    features = session_state.get("features")
    code_ctx = session_state.get("code_context")

    has_something = False

    # DataFrame context
    if df_ctx:
        has_something = True
        print(f"\n{C.CYAN}{C.BOLD}  ┌─ DataFrame Context ────────────────────────────────┐{C.RESET}")
        print(f"  {C.WHITE}│{C.RESET}  Columns:    {C.WHITE}{df_ctx.get('num_columns', '?')}{C.RESET}")
        print(f"  {C.WHITE}│{C.RESET}  Partitions: {C.WHITE}{df_ctx.get('num_partitions', '?')}{C.RESET}")
        size_mb = df_ctx.get("estimated_size_mb")
        if size_mb is not None:
            print(f"  {C.WHITE}│{C.RESET}  Est. Size:  {C.WHITE}{size_mb:.4f} MB{C.RESET}")
        print(f"  {C.WHITE}│{C.RESET}  Plan:       {C.GREEN}Captured{C.RESET}")
        # Quick schema summary
        schema = df_ctx.get("schema")
        if isinstance(schema, list) and schema:
            print(f"  {C.DIM}│  {'─' * 47}{C.RESET}")
            print(f"  {C.WHITE}│{C.RESET}  {C.GRAY}{'Column':<25} {'Type':<22}{C.RESET}")
            for field in schema[:10]:  # Cap at 10 to avoid flooding
                print(f"  {C.WHITE}│{C.RESET}  {C.WHITE}{field['name']:<25}{C.RESET} {C.GRAY}{field['type']:<22}{C.RESET}")
            if len(schema) > 10:
                print(f"  {C.WHITE}│{C.RESET}  {C.DIM}... and {len(schema) - 10} more columns{C.RESET}")
        print(f"{C.CYAN}  └────────────────────────────────────────────────────┘{C.RESET}")

    # DAG metrics (legacy)
    if features and isinstance(features, list):
        has_something = True
        print(f"\n{C.CYAN}{C.BOLD}  ┌─ DAG Metrics ────────────────────────────────────────┐{C.RESET}")
        print(f"  {C.WHITE}│  {'Stage':<12} {'Skew Ratio':<15} {'Status':<20}  │{C.RESET}")
        print(f"  {C.DIM}│  {'─' * 12} {'─' * 15} {'─' * 20}  │{C.RESET}")
        for f in features:
            stage = str(f.get("stage_id", "?"))
            ratio = f.get("skew_ratio", 0)
            if ratio > 5.0:
                status = f"{C.RED}● Critical Skew{C.RESET}"
                ratio_str = f"{C.RED}{ratio:.2f}{C.RESET}"
            elif ratio > 3.0:
                status = f"{C.ORANGE}● High Skew{C.RESET}"
                ratio_str = f"{C.ORANGE}{ratio:.2f}{C.RESET}"
            elif ratio > 1.5:
                status = f"{C.YELLOW}● Moderate{C.RESET}"
                ratio_str = f"{C.YELLOW}{ratio:.2f}{C.RESET}"
            else:
                status = f"{C.GREEN}● Healthy{C.RESET}"
                ratio_str = f"{C.GREEN}{ratio:.2f}{C.RESET}"
            print(f"  {C.WHITE}│  {stage:<12} {ratio_str:<24} {status:<29}  │{C.RESET}")
        print(f"{C.CYAN}  └────────────────────────────────────────────────────┘{C.RESET}")

    # Code context
    if code_ctx:
        has_something = True
        _print_code_block(code_ctx, title="Captured PySpark Code")

    if not has_something:
        print(f"\n  {C.YELLOW}⚠  No context available for this session.{C.RESET}")


def _print_plan(session_state):
    """Print the Catalyst execution plan."""
    df_ctx = session_state.get("df_context")
    if not df_ctx:
        print(f"\n  {C.YELLOW}⚠  No DataFrame was provided — no execution plan available.{C.RESET}")
        return
    plan = df_ctx.get("execution_plan", "Not captured")
    if plan == "Could not extract execution plan":
        print(f"\n  {C.YELLOW}⚠  Could not extract the execution plan.{C.RESET}")
        return
    _print_code_block(plan, title="Catalyst Execution Plan")


def _print_schema(session_state):
    """Print the DataFrame schema."""
    df_ctx = session_state.get("df_context")
    if not df_ctx:
        print(f"\n  {C.YELLOW}⚠  No DataFrame was provided — no schema available.{C.RESET}")
        return
    schema = df_ctx.get("schema")
    if not isinstance(schema, list):
        print(f"\n  {C.YELLOW}⚠  {schema}{C.RESET}")
        return

    print(f"\n{C.CYAN}{C.BOLD}  ┌─ DataFrame Schema ─────────────────────────────────┐{C.RESET}")
    print(f"  {C.WHITE}│  {'Column':<25} {'Type':<18} {'Nullable':<8} │{C.RESET}")
    print(f"  {C.DIM}│  {'─' * 25} {'─' * 18} {'─' * 8} │{C.RESET}")
    for field in schema:
        nullable = f"{C.GREEN}✔{C.RESET}" if field["nullable"] else f"{C.RED}✖{C.RESET}"
        print(f"  {C.WHITE}│  {field['name']:<25} {C.GRAY}{field['type']:<18}{C.RESET} {nullable:<17} {C.WHITE}│{C.RESET}")
    print(f"{C.CYAN}  └────────────────────────────────────────────────────┘{C.RESET}")


def _print_code_block(code, title="Code"):
    """Render a formatted code block in the terminal."""
    print(f"\n  {C.BLUE}{C.BOLD}┌─ {title} {'─' * max(1, 48 - len(title))}┐{C.RESET}")
    for line in code.strip().split("\n"):
        print(f"  {C.BLUE}│{C.RESET}  {C.WHITE}{line}{C.RESET}")
    print(f"  {C.BLUE}└{'─' * 52}┘{C.RESET}")


def _print_thinking():
    """Print a thinking indicator."""
    print(f"\n  {C.MAGENTA}◇{C.RESET} {C.DIM}Analyzing with Gemini...{C.RESET}")


def _print_response(text, msg_num):
    """Render the AI response with formatting."""
    print(f"\n  {C.ORANGE}{C.BOLD}┌─ OptiSpark #{msg_num} {'─' * 38}┐{C.RESET}")

    # Process the response line by line
    in_code_block = False
    lines = text.strip().split("\n")

    for line in lines:
        stripped = line.strip()

        # Toggle code block state
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                lang = stripped[3:].strip() or "pyspark"
                print(f"  {C.ORANGE}│{C.RESET}")
                print(f"  {C.ORANGE}│{C.RESET}  {C.BLUE}{C.DIM}── {lang} ──{C.RESET}")
            else:
                print(f"  {C.ORANGE}│{C.RESET}  {C.BLUE}{C.DIM}──────────{C.RESET}")
                print(f"  {C.ORANGE}│{C.RESET}")
            continue

        if in_code_block:
            print(f"  {C.ORANGE}│{C.RESET}  {C.GREEN}  {line}{C.RESET}")
        elif stripped.startswith("**") and stripped.endswith("**"):
            # Bold headers
            clean = stripped.strip("*").strip()
            print(f"  {C.ORANGE}│{C.RESET}  {C.WHITE}{C.BOLD}{clean}{C.RESET}")
        elif stripped.startswith("- ") or stripped.startswith("• "):
            # Bullet points
            print(f"  {C.ORANGE}│{C.RESET}  {C.CYAN}  •{C.RESET} {C.WHITE}{stripped[2:]}{C.RESET}")
        elif stripped.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            # Numbered lists
            num, rest = stripped.split(".", 1)
            print(f"  {C.ORANGE}│{C.RESET}  {C.CYAN}  {num}.{C.RESET}{C.WHITE}{rest}{C.RESET}")
        else:
            # Wrap long lines for readability
            wrapped = textwrap.fill(line, width=70, subsequent_indent="    ")
            for wline in wrapped.split("\n"):
                print(f"  {C.ORANGE}│{C.RESET}  {C.WHITE}{wline}{C.RESET}")

    print(f"  {C.ORANGE}└{'─' * 52}┘{C.RESET}")


def _print_error(error_msg):
    """Print a formatted error message."""
    print(f"\n  {C.RED}{C.BOLD}┌─ Error ───────────────────────────────────────────┐{C.RESET}")
    print(f"  {C.RED}│{C.RESET}  {error_msg}")
    print(f"  {C.RED}│{C.RESET}  {C.GRAY}Try rephrasing your question or type /help.{C.RESET}")
    print(f"  {C.RED}└{'─' * 52}┘{C.RESET}")


def _print_goodbye(session_state):
    """Print session summary on exit."""
    elapsed = datetime.now() - session_state["start_time"]
    mins = int(elapsed.total_seconds() // 60)
    secs = int(elapsed.total_seconds() % 60)
    msgs = session_state["message_count"]

    print(f"""
{C.DIM}  {'─' * 55}{C.RESET}
{C.CYAN}{C.BOLD}  ✦ Session Complete{C.RESET}
{C.GRAY}    Messages: {C.WHITE}{msgs}{C.GRAY} · Duration: {C.WHITE}{mins}m {secs}s{C.RESET}
{C.GRAY}    Thank you for using {C.CYAN}OptiSpark{C.GRAY}. ⚡{C.RESET}
{C.DIM}  {'─' * 55}{C.RESET}
""")


def _clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")

def _print_benchmark_results(results):
    """Render a premium ANSI table for the benchmark results."""
    from optispark.agent import _Colors as C
    
    if results.get("status") == "error":
        print(f"\n  {C.RED}{C.BOLD}┌─ Benchmark Failed ──────────────────────────────────────────────┐{C.RESET}")
        print(f"  {C.RED}│{C.RESET}  {results.get('message', 'Unknown Error')}")
        print(f"  {C.RED}└{'─' * 65}┘{C.RESET}")
        return
        
    orig_time = results["original_time_sec"]
    fixed_time = results["fixed_time_sec"]
    pct = results["improvement_pct"]
    
    # Determine color rendering
    pct_color = C.GREEN if pct > 0 else C.RED
    trend = "🚀 Faster!" if pct > 0 else "🐢 Slower"
    
    print(f"\n  {C.CYAN}{C.BOLD}┌─ 📊 Dry-Run Benchmark Results (0.1% Sample) ──────────────────────┐{C.RESET}")
    print(f"  {C.CYAN}│{C.RESET}  Original Plan Execution Time : {C.WHITE}{orig_time:>6.2f}s{C.RESET}")
    print(f"  {C.CYAN}│{C.RESET}  AI-Fixed Plan Execution Time : {C.WHITE}{fixed_time:>6.2f}s{C.RESET}")
    print(f"  {C.CYAN}├───────────────────────────────────────────────────────────────────┤{C.RESET}")
    print(f"  {C.CYAN}│{C.RESET}  Net Improvement               : {pct_color}{C.BOLD}{pct:>6.2f}%{C.RESET} {trend}")
    print(f"  {C.CYAN}└{'─' * 67}┘{C.RESET}")