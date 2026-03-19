from setuptools import setup, find_packages

setup(
    name="optispark",
    version="0.1.0",  # MVP version
    author="Your Name",
    description="An LLM-powered autonomous Spark optimization agent.",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyspark>=3.0.0",
        "zstandard>=0.22.0",
        "google-generativeai>=0.4.0"
    ],
    entry_points={
        "console_scripts": [
            "optispark=optispark.cli:main",
        ],
    },
    python_requires=">=3.9",
)