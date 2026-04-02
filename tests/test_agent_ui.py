import pytest
from datetime import datetime
from optispark.agent import (
    _print_banner, _print_ready_prompt, _print_help, _print_context,
    _print_plan, _print_schema, _print_code_block, _print_thinking,
    _print_response, _print_error, _print_goodbye, _clear_screen, _print_benchmark_results
)

def test_ui_renders(capsys):
    _print_banner()
    _print_ready_prompt()
    _print_help()
    
    session_state = {
        "start_time": datetime.now(),
        "message_count": 1,
        "df_context": {
            "num_columns": 2, "num_partitions": 4, "estimated_size_mb": 1.5,
            "execution_plan": "Scan...", "schema": [{"name": "id", "type": "int", "nullable": False}] * 15
        },
        "features": [{"stage_id": 1, "skew_ratio": 6.0}, {"stage_id": 2, "skew_ratio": 4.0}, {"stage_id": 3, "skew_ratio": 2.0}, {"stage_id": 4, "skew_ratio": 1.0}],
        "code_context": "code"
    }
    
    _print_context(session_state)
    _print_plan(session_state)
    _print_schema(session_state)
    _print_code_block("code line")
    _print_thinking()
    
    # test markdown rendering
    _print_response("**bold**\n- bullet\n1. num\n```python\ndf=1\n```\nlong text", 1)
    
    _print_error("bad")
    _print_goodbye(session_state)
    
    _print_benchmark_results({"status": "success", "original_time_sec": 1, "fixed_time_sec": 0.5, "improvement_pct": 50})
    _print_benchmark_results({"status": "success", "original_time_sec": 1, "fixed_time_sec": 2, "improvement_pct": -50})
    _print_benchmark_results({"status": "error", "message": "Failed"})

    # test empty context
    _print_context({})
    _print_schema({})
    _print_plan({})

def test_clear_screen():
    _clear_screen()
