import pytest
import os

@pytest.fixture(scope="session")
def spark():
    """Build a reusable local SparkSession for tests.
    
    If PySpark/Java is not available (e.g. lightweight CI), skip tests 
    that require a live Spark session.
    """
    try:
        from pyspark.sql import SparkSession
        
        # Force pyspark to use the correct Python
        os.environ["PYSPARK_PYTHON"] = os.sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = os.sys.executable
        
        spark_session = SparkSession.builder \
            .master("local[2]") \
            .appName("OptiSpark_Tests") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.ui.enabled", "false") \
            .config("spark.driver.host", "localhost") \
            .getOrCreate()

        yield spark_session
        spark_session.stop()
    except Exception as e:
        pytest.skip(f"PySpark not available: {e}")
