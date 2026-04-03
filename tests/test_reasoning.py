import pytest
from unittest.mock import patch, MagicMock
from optispark.reasoning import ReasoningEngine, _RemoteChatSession


@pytest.fixture
def mock_post():
    with patch("optispark.reasoning.requests.post") as mock:
        yield mock


def test_start_chat_success(mock_post):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"session_id": "abc-123", "model_used": "gemma-4-31b-it"}
    mock_post.return_value = resp

    engine = ReasoningEngine(server_url="http://localhost:8000")
    chat = engine.start_chat({"dataframe": {"num_columns": 5}})

    assert isinstance(chat, _RemoteChatSession)
    assert chat.session_id == "abc-123"
    assert engine.model_id == "gemma-4-31b-it"


def test_start_chat_connection_error(mock_post):
    import requests
    mock_post.side_effect = requests.ConnectionError("Connection refused")

    engine = ReasoningEngine(server_url="http://localhost:9999")
    with pytest.raises(RuntimeError) as exc:
        engine.start_chat({})
    assert "Could not connect" in str(exc.value)


def test_start_chat_server_error(mock_post):
    resp = MagicMock()
    resp.status_code = 503
    resp.json.return_value = {"detail": "All models failed"}
    mock_post.return_value = resp

    engine = ReasoningEngine(server_url="http://localhost:8000")
    with pytest.raises(RuntimeError) as exc:
        engine.start_chat({})
    assert "503" in str(exc.value)


def test_send_message(mock_post):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"text": "Use salted join", "session_id": "abc-123"}
    mock_post.return_value = resp

    chat = _RemoteChatSession(session_id="abc-123", server_url="http://localhost:8000")
    result = chat.send_message("fix my skew")

    assert result.text == "Use salted join"


def test_send_message_error(mock_post):
    resp = MagicMock()
    resp.status_code = 500
    resp.json.return_value = {"detail": "Model error"}
    mock_post.return_value = resp

    chat = _RemoteChatSession(session_id="abc-123", server_url="http://localhost:8000")
    with pytest.raises(RuntimeError):
        chat.send_message("fix it")


def test_diagnose(mock_post):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"text": "Skew detected", "model_used": "gemma-4-31b-it"}
    mock_post.return_value = resp

    engine = ReasoningEngine(server_url="http://localhost:8000")
    result = engine.diagnose([{"stage_id": 1}])
    assert result == "Skew detected"

    # Verify use_fallback is False for diagnose
    call_args = mock_post.call_args
    assert call_args[1]["json"]["use_fallback"] is False


def test_generate_fix_uses_fallback(mock_post):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"text": "df_opt = df", "model_used": "gemini-3-flash-preview"}
    mock_post.return_value = resp

    engine = ReasoningEngine(server_url="http://localhost:8000")
    result = engine.generate_fix([{"stage_id": 1}])
    assert result == "df_opt = df"

    # Verify use_fallback is True for generate_fix (complex task)
    call_args = mock_post.call_args
    assert call_args[1]["json"]["use_fallback"] is True


def test_generate_connection_error(mock_post):
    import requests
    mock_post.side_effect = requests.ConnectionError("Timeout")

    engine = ReasoningEngine(server_url="http://localhost:8000")
    with pytest.raises(RuntimeError) as exc:
        engine._generate("test")
    assert "Could not connect" in str(exc.value)


def test_generate_server_error(mock_post):
    resp = MagicMock()
    resp.status_code = 503
    resp.json.return_value = {"detail": "All models failed"}
    mock_post.return_value = resp

    engine = ReasoningEngine(server_url="http://localhost:8000")
    with pytest.raises(RuntimeError):
        engine._generate("test")


def test_default_server_url():
    engine = ReasoningEngine()
    assert "onrender.com" in engine.server_url
