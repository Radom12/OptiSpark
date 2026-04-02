import pytest
from unittest.mock import patch, MagicMock
from optispark.reasoning import ReasoningEngine

@pytest.fixture
def mock_genai():
    with patch("optispark.reasoning.genai") as mock_genai:
        yield mock_genai

def test_diagnose(mock_genai):
    client = MagicMock()
    mock_genai.Client.return_value = client
    resp = MagicMock()
    resp.text = "Mocked Diagnosis"
    client.models.generate_content.return_value = resp
    
    engine = ReasoningEngine("fake_key")
    res = engine.diagnose([{"stage_id": 1}])
    assert res == "Mocked Diagnosis"

def test_generate_fix(mock_genai):
    client = MagicMock()
    mock_genai.Client.return_value = client
    resp = MagicMock()
    resp.text = "df_opt = df"
    client.models.generate_content.return_value = resp
    
    engine = ReasoningEngine("fake_key")
    res = engine.generate_fix([{"stage_id": 1}])
    assert res == "df_opt = df"

def test_start_chat_success(mock_genai):
    client = MagicMock()
    mock_genai.Client.return_value = client
    chat = MagicMock()
    client.chats.create.return_value = chat
    
    engine = ReasoningEngine("fake_key")
    c = engine.start_chat({"dataframe": {"num_columns": 5}})
    assert c == chat
    
def test_start_chat_fallback_error(mock_genai):
    client = MagicMock()
    mock_genai.Client.return_value = client
    client.chats.create.side_effect = Exception("API Error")
    
    engine = ReasoningEngine("fake_key")
    with pytest.raises(RuntimeError) as e:
        engine.start_chat({})
    assert "All fallback models failed" in str(e.value)
    
def test_generate_fallback_error(mock_genai):
    client = MagicMock()
    mock_genai.Client.return_value = client
    client.models.generate_content.side_effect = Exception("API Error")
    
    engine = ReasoningEngine("fake_key")
    with pytest.raises(RuntimeError):
        engine._generate("hi")

def test_build_context_injection():
    engine = ReasoningEngine("fake_key")
    
    # 1. Full dataframe context
    ctx1 = {
        "dataframe": {
            "schema": [{"name": "id", "type": "int", "nullable": False}],
            "num_columns": 1, "num_partitions": 2, "estimated_size_mb": 1.5, "estimated_size_bytes": 1500,
            "execution_plan": "Scan...", "logical_plan": "Relation...",
            "spark_conf": {"spark.sql.shuffle.partitions": "200"}
        },
        "dag_metrics": [{"skew_ratio": 6.0}], # forces critical skew msg
        "statement_text": "df = spark.range(10)"
    }
    inj1 = engine._build_context_injection(ctx1)
    assert "DataFrame Schema" in inj1
    assert "Estimated Size" in inj1
    assert "Execution Plan" in inj1
    assert "Spark Configuration" in inj1
    assert "DAG Stage Metrics" in inj1
    assert "PySpark Code Executed" in inj1
    
    # 2. Empty context
    inj2 = engine._build_context_injection({})
    assert "No DataFrame or metrics context available" in inj2
    
    # 3. Alternate forms
    ctx3 = {
        "dataframe": {
            "schema": "Just string schema"
        },
        "dag_metrics": {"raw_dict_metrics": 1}
    }
    inj3 = engine._build_context_injection(ctx3)
    assert "Just string schema" in inj3
