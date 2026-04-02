import pytest
from unittest.mock import patch, MagicMock
from optispark.cli import main
import os

@patch("optispark.cli.OptiSpark")
@patch("sys.argv", ["optispark", "analyze", "--log-dir", "/dummy"])
def test_cli_analyze(mock_optispark):
    agent_instance = MagicMock()
    mock_optispark.return_value = agent_instance
    os.environ["GEMINI_API_KEY"] = "fake"
    main()
    agent_instance.optimize.assert_called_once_with(target_df=None)

@patch("optispark.cli.OptiSpark")
@patch("sys.argv", ["optispark", "chat", "--log-dir", "/dummy"])
def test_cli_chat(mock_optispark):
    agent_instance = MagicMock()
    mock_optispark.return_value = agent_instance
    os.environ["GEMINI_API_KEY"] = "fake"
    main()
    agent_instance.chat.assert_called_once()
    
@patch("sys.argv", ["optispark", "chat", "--log-dir", "/dummy"])
@patch("builtins.print")
def test_cli_missing_key(mock_print):
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    main()
    mock_print.assert_any_call("❌ Error: GEMINI_API_KEY environment variable not set.")
    os.environ["GEMINI_API_KEY"] = "mocked_for_tests" # Restore for others
