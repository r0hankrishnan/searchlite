from setuptools import setup, find_packages

setup(
    name="searchlite",
    version="0.2.0",
    author="Rohan Krishnan",
    description="A simple Python package that allows semantic search with simple syntax on small text data sets.",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0",
        "tabulate>=0.8.0",
        "numpy<2",
        "pandas>=1.0"
    ],
    extras_require={
    "dev": ["pytest"],
    "sentence_transformers": ["sentence-transformers>=2.2.0"],
    "ollama": ["ollama"]
    },
    python_requires=">=3.7",
    license="MIT",
)
