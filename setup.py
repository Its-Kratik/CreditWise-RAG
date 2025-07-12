# setup.py
from setuptools import setup, find_packages

setup(
    name="rag_math_engine",
    version="0.1.0",
    description="Smart math-aware RAG query engine for tabular data",
    author="Kratik Jain",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "spacy",
        "scikit-learn"
    ],
    python_requires=">=3.7",
)
