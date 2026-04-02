import pytest
from pyspark.sql import SparkSession
import os

@pytest.fixture(scope="session")
def spark():
    """Build a reusable local SparkSession for tests."""
    # Set dummy API key to pass the CLI check inherently if loaded via dotenv
    os.environ["GEMINI_API_KEY"] = "mocked_for_tests"
    
    spark_session = SparkSession.builder \
        .master("local[2]") \
        .appName("OptiSpark_Tests") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    
    yield spark_session
    spark_session.stop()
