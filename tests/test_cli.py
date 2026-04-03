import pytest
from unittest.mock import patch, MagicMock
from optispark.cli import main

@patch("optispark.cli.OptiSpark")
@patch("sys.argv", ["optispark", "analyze", "--log-dir", "/dummy"])
def test_cli_analyze(mock_optispark):
    agent_instance = MagicMock()
    mock_optispark.return_value = agent_instance
    main()
    agent_instance.optimize.assert_called_once_with(target_df=None)

@patch("optispark.cli.OptiSpark")
@patch("sys.argv", ["optispark", "chat", "--log-dir", "/dummy"])
def test_cli_chat(mock_optispark):
    agent_instance = MagicMock()
    mock_optispark.return_value = agent_instance
    main()
    agent_instance.chat.assert_called_once()

@patch("optispark.cli.OptiSpark")
@patch("sys.argv", ["optispark", "chat", "--log-dir", "/dummy", "--server-url", "http://custom:8000"])
def test_cli_custom_server_url(mock_optispark):
    agent_instance = MagicMock()
    mock_optispark.return_value = agent_instance
    main()
    mock_optispark.assert_called_once_with(log_dir="/dummy", server_url="http://custom:8000")
