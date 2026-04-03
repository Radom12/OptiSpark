import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from optispark.agent import OptiSpark
import os


def _make_mock_df(columns=None):
    """Create a mock DataFrame that behaves like a PySpark DataFrame."""
    mock_df = MagicMock()
    mock_df.columns = columns or ["id"]
    mock_df.schema = MagicMock()
    mock_df.schema.fields = [MagicMock(name="id", dataType=MagicMock(__str__=lambda s: "LongType"), nullable=False)]
    mock_df.schema.jsonValue.return_value = {"fields": [{"name": "id", "type": "long", "nullable": False}]}
    mock_df.rdd.getNumPartitions.return_value = 2
    mock_df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes.return_value = 1024
    mock_df.explain = MagicMock(return_value=None)
    mock_df.sparkSession = MagicMock()
    mock_df.sparkSession.conf.get = MagicMock(return_value="200")
    return mock_df


@patch("optispark.agent.ReasoningEngine")
@patch("optispark.agent.extract_features_from_logs")
@patch("optispark.agent.extract_features_from_system_tables")
def test_optimize_success(mock_sys, mock_logs, mock_engine):
    mock_sys.return_value = [{"stage_id": 1}]
    engine_instance = MagicMock()
    engine_instance.diagnose.return_value = "Bad Join"
    engine_instance.generate_fix.return_value = "optimized_df = df"
    mock_engine.return_value = engine_instance

    agent = OptiSpark(log_dir="/dev/null")

    with patch("optispark.agent.validate_safety") as mock_safety:
        mock_safety.return_value = (True, "Safe")
        agent.optimize(spark=MagicMock(), query_id="123", target_df=MagicMock())

        engine_instance.diagnose.assert_called_once()
        engine_instance.generate_fix.assert_called_once()

@patch("optispark.agent.ReasoningEngine")
@patch("optispark.agent.extract_features_from_system_tables")
def test_optimize_no_metrics(mock_sys, mock_engine):
    mock_sys.return_value = None

    agent = OptiSpark()
    agent.optimize(spark=MagicMock(), query_id="123")

    mock_engine.return_value.diagnose.assert_not_called()

@patch("optispark.agent.ReasoningEngine")
@patch("optispark.agent.extract_features_from_system_tables")
def test_optimize_safety_blocked(mock_sys, mock_engine):
    mock_sys.return_value = [{"stage_id": 1}]
    engine_instance = MagicMock()
    engine_instance.diagnose.return_value = "Diagnosis"
    engine_instance.generate_fix.return_value = "code"
    mock_engine.return_value = engine_instance

    agent = OptiSpark()

    with patch("optispark.agent.validate_safety") as mock_safety:
        mock_safety.return_value = (False, "OOM risk")
        agent.optimize(spark=MagicMock(), query_id="123", target_df=MagicMock())

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input")
def test_chat_exit(mock_input, mock_engine):
    mock_input.side_effect = ["exit"]

    chat_session = MagicMock()
    mock_engine.return_value.start_chat.return_value = chat_session

    agent = OptiSpark()
    df = _make_mock_df()
    returned_df = agent.chat(df=df)

    assert returned_df == df

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input")
def test_chat_interaction_and_execution(mock_input, mock_engine):
    mock_input.side_effect = ["fix this", "y", "exit"]

    chat_session = MagicMock()
    resp = MagicMock()
    resp.text = """Some analysis.
```python
optimized_df = df.withColumn('opt', F.lit(1))
```
"""
    chat_session.send_message.return_value = resp
    mock_engine.return_value.start_chat.return_value = chat_session

    agent = OptiSpark()
    df = _make_mock_df()
    returned_df = agent.chat(df=df)
    # Should have attempted execution (the exec will work on mock)

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["/help", "/m", "/p", "/s", "/clear", "exit"])
@patch("optispark.agent._clear_screen")
def test_chat_commands(mock_clear, mock_input, mock_engine):
    agent = OptiSpark()
    agent.chat(df=_make_mock_df())
    mock_clear.assert_called()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["fix", "y", "/b", "exit"])
@patch("optispark.benchmark.run_benchmark")
def test_chat_benchmark(mock_benchmark, mock_input, mock_engine):
    chat_session = MagicMock()
    resp = MagicMock()
    resp.text = "```python\noptimized_df = df\n```"
    chat_session.send_message.return_value = resp
    mock_engine.return_value.start_chat.return_value = chat_session
    mock_benchmark.return_value = {"status": "success", "original_time_sec": 1, "fixed_time_sec": 0.5, "improvement_pct": 50}

    agent = OptiSpark()
    agent.chat(df=_make_mock_df())
    mock_benchmark.assert_called_once()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input")
def test_chat_keyboard_interrupt(mock_input, mock_engine):
    mock_input.side_effect = KeyboardInterrupt()
    agent = OptiSpark()
    agent.chat()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["exit"])
def test_agent_introspect_error_handling(mock_input, mock_engine):
    class BadDF:
        @property
        def schema(self):
            raise Exception("Schema failed")
        @property
        def _jdf(self):
            raise Exception("Catalyst failed")
        @property
        def rdd(self):
            class BadRDD:
                def getNumPartitions(self):
                    raise Exception("Partition failed")
            return BadRDD()
        def explain(self, mode="extended"):
            raise Exception("Explain failed")

    agent = OptiSpark()
    bdf = BadDF()
    res = agent.chat(df=bdf)
    assert res == bdf

@patch("optispark.agent.ReasoningEngine")
@patch("optispark.agent.extract_features_from_system_tables")
@patch("optispark.agent.OptiSpark._fetch_statement_text")
@patch("builtins.input", side_effect=["exit"])
def test_agent_chat_legacy_path(mock_input, mock_fetch, mock_sys, mock_engine):
    mock_sys.return_value = [{"stage_id": 1, "skew_ratio": 4.0}]
    mock_fetch.return_value = "code"
    agent = OptiSpark()
    agent.chat(spark=MagicMock(), query_id="123")
    mock_sys.assert_called_once()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["exit"])
def test_agent_chat_log_dir_path(mock_input, mock_engine):
    with patch("optispark.agent.extract_features_from_logs") as mock_logs:
        mock_logs.return_value = [{"stage_id": 1, "skew_ratio": 2.0}]
        agent = OptiSpark(log_dir="/fake/logs")
        agent.chat()
        mock_logs.assert_called_once()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["exit"])
def test_agent_chat_session_init_failure(mock_input, mock_engine):
    mock_engine.return_value.start_chat.side_effect = RuntimeError("Backend down")
    agent = OptiSpark()
    result = agent.chat()
    assert result is None

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["fix this", "y", "exit"])
def test_chat_exec_failure_self_heal(mock_input, mock_engine):
    chat_session = MagicMock()
    resp = MagicMock()
    resp.text = "```python\noptimized_df = undefined_var\n```"
    correction = MagicMock()
    correction.text = "```python\noptimized_df = df.withColumn('fixed', F.lit(1))\n```"
    chat_session.send_message.side_effect = [resp, correction]
    mock_engine.return_value.start_chat.return_value = chat_session

    agent = OptiSpark()
    agent.chat(df=_make_mock_df())

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["ask something", "exit"])
def test_chat_generic_exception(mock_input, mock_engine):
    chat_session = MagicMock()
    chat_session.send_message.side_effect = [Exception("Random error"), MagicMock(text="ok")]
    mock_engine.return_value.start_chat.return_value = chat_session

    agent = OptiSpark()
    agent.chat()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["/b", "exit"])
def test_chat_benchmark_no_code(mock_input, mock_engine):
    agent = OptiSpark()
    agent.chat()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["", "exit"])
def test_chat_empty_input(mock_input, mock_engine):
    agent = OptiSpark()
    agent.chat()
