import pytest
from unittest.mock import patch, MagicMock
from optispark.agent import OptiSpark
import os

@patch("optispark.agent.ReasoningEngine")
@patch("optispark.agent.extract_features_from_logs")
@patch("optispark.agent.extract_features_from_system_tables")
def test_optimize_success(mock_sys, mock_logs, mock_engine):
    # Setup mocks
    mock_sys.return_value = [{"stage_id": 1}]
    engine_instance = MagicMock()
    engine_instance.diagnose.return_value = "Bad Join"
    engine_instance.generate_fix.return_value = "df_opt = df"
    mock_engine.return_value = engine_instance
    
    os.environ["GEMINI_API_KEY"] = "fake"
    agent = OptiSpark(log_dir="/dev/null")
    
    with patch("optispark.agent.validate_safety") as mock_safety:
        mock_safety.return_value = (True, "Safe")
        # Call optimize
        agent.optimize(spark=MagicMock(), query_id="123", target_df=MagicMock())
        
        engine_instance.diagnose.assert_called_once()
        engine_instance.generate_fix.assert_called_once()

@patch("optispark.agent.ReasoningEngine")
@patch("optispark.agent.extract_features_from_system_tables")
def test_optimize_no_metrics(mock_sys, mock_engine):
    mock_sys.return_value = None
    
    agent = OptiSpark(api_key="fake")
    # Should abort early
    agent.optimize(spark=MagicMock(), query_id="123")
    
    mock_engine.return_value.diagnose.assert_not_called()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input")
def test_chat_exit(mock_input, mock_engine, spark):
    # Setup REPL sequence: just type 'exit'
    mock_input.side_effect = ["exit"]
    
    chat_session = MagicMock()
    mock_engine.return_value.start_chat.return_value = chat_session
    
    agent = OptiSpark(api_key="fake")
    df = spark.range(5)
    returned_df = agent.chat(df=df)
    
    assert returned_df == df

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input")
def test_chat_interaction_and_execution(mock_input, mock_engine, spark):
    # Simulate user sending a message, accepting the code, then exiting
    mock_input.side_effect = ["fix this", "y", "exit"]
    
    chat_session = MagicMock()
    resp = MagicMock()
    # Provide an executable block
    resp.text = """Some analysis.\n```python\ndf_opt = df.withColumn('opt', F.lit(1))\n```\n"""
    chat_session.send_message.return_value = resp
    mock_engine.return_value.start_chat.return_value = chat_session
    
    agent = OptiSpark(api_key="fake")
    df = spark.range(5)
    
    # Capture printed output to avoid clutter, just check if it returns df_opt
    returned_df = agent.chat(df=df)
    
    assert "opt" in returned_df.columns

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["/help", "/m", "/p", "/s", "/clear", "exit"])
@patch("optispark.agent._clear_screen")
def test_chat_commands(mock_clear, mock_input, mock_engine, spark):
    # Test UI commands just to run through them
    agent = OptiSpark(api_key="fake")
    agent.chat(df=spark.range(1))
    mock_clear.assert_called()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input", side_effect=["fix", "y", "/b", "exit"])
@patch("optispark.agent.run_benchmark")
def test_chat_benchmark(mock_benchmark, mock_input, mock_engine, spark):
    # Test /benchmark integration
    chat_session = MagicMock()
    resp = MagicMock()
    resp.text = "```python\ndf_opt = df\n```"
    chat_session.send_message.return_value = resp
    mock_engine.return_value.start_chat.return_value = chat_session
    mock_benchmark.return_value = {"status": "success", "original_time_sec": 1, "fixed_time_sec": 0.5, "improvement_pct": 50}
    
    agent = OptiSpark(api_key="fake")
    agent.chat(df=spark.range(1))
    mock_benchmark.assert_called_once()

@patch("optispark.agent.ReasoningEngine")
@patch("builtins.input")
def test_chat_keyboard_interrupt(mock_input, mock_engine):
    mock_input.side_effect = KeyboardInterrupt()
    agent = OptiSpark(api_key="fake")
    agent.chat() # No df, general mode
    # Should exit cleanly
