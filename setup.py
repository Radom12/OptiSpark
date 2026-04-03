import os
from setuptools import setup, find_packages

def read_readme():
    try:
        if os.path.exists("README.md"):
            with open("README.md", "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ""

setup(
    name="optispark",
    version="0.2.0",
    author="Your Name",
    description="An LLM-powered autonomous Spark optimization agent.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyspark>=3.0.0",
        "zstandard>=0.22.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "optispark=optispark.cli:main",
        ],
    },
    python_requires=">=3.9",
)