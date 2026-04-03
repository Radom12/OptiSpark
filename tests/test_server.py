import pytest
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure the project root is on sys.path so 'server' package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set the API key before importing the server module
os.environ["GEMINI_API_KEY"] = "test_server_key"

from fastapi.testclient import TestClient
from server.main import app, chat_sessions, _build_context_injection, _cleanup_expired_sessions

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["primary_model"] == "gemma-3-27b"


@patch("server.main.client")
def test_start_chat_success(mock_genai_client):
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat

    resp = client.post("/api/v1/chat/start", json={
        "combined_context": {"dataframe": {"num_columns": 3}}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["model_used"] == "gemma-3-27b"


@patch("server.main.client")
def test_start_chat_all_models_fail(mock_genai_client):
    mock_genai_client.chats.create.side_effect = Exception("Quota exceeded")

    resp = client.post("/api/v1/chat/start", json={
        "combined_context": {}
    })
    assert resp.status_code == 503
    assert "All models failed" in resp.json()["detail"]


@patch("server.main.client")
def test_send_message_success(mock_genai_client):
    # First create a session
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat

    start_resp = client.post("/api/v1/chat/start", json={
        "combined_context": {}
    })
    session_id = start_resp.json()["session_id"]

    # Now send a message
    mock_response = MagicMock()
    mock_response.text = "Use a salted join to fix skew."
    mock_chat.send_message.return_value = mock_response

    msg_resp = client.post("/api/v1/chat/message", json={
        "session_id": session_id,
        "message": "Fix my join"
    })
    assert msg_resp.status_code == 200
    assert msg_resp.json()["text"] == "Use a salted join to fix skew."


def test_send_message_invalid_session():
    resp = client.post("/api/v1/chat/message", json={
        "session_id": "nonexistent",
        "message": "hello"
    })
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


@patch("server.main.client")
def test_send_message_model_error(mock_genai_client):
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat

    start_resp = client.post("/api/v1/chat/start", json={"combined_context": {}})
    session_id = start_resp.json()["session_id"]

    mock_chat.send_message.side_effect = Exception("Context too long")

    msg_resp = client.post("/api/v1/chat/message", json={
        "session_id": session_id,
        "message": "fix"
    })
    assert msg_resp.status_code == 500
    assert "Model error" in msg_resp.json()["detail"]


@patch("server.main.client")
def test_generate_success(mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = "df_opt = df.repartition(200)"
    mock_genai_client.models.generate_content.return_value = mock_response

    resp = client.post("/api/v1/generate", json={
        "prompt": "Fix shuffle",
        "use_fallback": False
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "df_opt = df.repartition(200)"
    assert data["model_used"] == "gemma-3-27b"


@patch("server.main.client")
def test_generate_with_fallback(mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = "complex fix"
    mock_genai_client.models.generate_content.return_value = mock_response

    resp = client.post("/api/v1/generate", json={
        "prompt": "Fix complex pipeline",
        "use_fallback": True
    })
    assert resp.status_code == 200
    # When use_fallback=True, Gemini models are tried first
    assert resp.json()["model_used"] == "gemini-2.0-flash"


@patch("server.main.client")
def test_generate_all_fail(mock_genai_client):
    mock_genai_client.models.generate_content.side_effect = Exception("Rate limited")

    resp = client.post("/api/v1/generate", json={
        "prompt": "Fix it",
        "use_fallback": False
    })
    assert resp.status_code == 503


def test_build_context_injection_full():
    ctx = {
        "dataframe": {
            "schema": [{"name": "id", "type": "int", "nullable": False}],
            "num_columns": 1, "num_partitions": 2,
            "estimated_size_mb": 1.5, "estimated_size_bytes": 1500,
            "execution_plan": "Scan...", "logical_plan": "Relation...",
            "spark_conf": {"spark.sql.shuffle.partitions": "200",
                           "spark.driver.memory": "4g",
                           "spark.executor.memory": "8g",
                           "spark.executor.cores": "4"}
        },
        "dag_metrics": [{"skew_ratio": 6.0}],
        "statement_text": "df = spark.range(10)"
    }
    result = _build_context_injection(ctx)
    assert "DataFrame Schema" in result
    assert "Estimated Size" in result
    assert "Execution Plan" in result
    assert "Optimized Logical Plan" in result
    assert "Spark Configuration" in result
    assert "DAG Stage Metrics" in result
    assert "PySpark Code Executed" in result


def test_build_context_injection_empty():
    result = _build_context_injection({})
    assert "No DataFrame or metrics context available" in result


def test_build_context_injection_string_schema():
    ctx = {"dataframe": {"schema": "raw string schema"}, "dag_metrics": {"raw": 1}}
    result = _build_context_injection(ctx)
    assert "raw string schema" in result


def test_cleanup_expired_sessions():
    import time
    chat_sessions["old-session"] = {
        "chat": MagicMock(),
        "model": "test",
        "created_at": time.time() - 7200  # 2 hours ago
    }
    _cleanup_expired_sessions()
    assert "old-session" not in chat_sessions
