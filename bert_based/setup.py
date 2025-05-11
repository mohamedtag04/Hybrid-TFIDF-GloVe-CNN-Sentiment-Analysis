from setuptools import setup, find_packages

setup(
    name="sentiment_hybrid",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "gensim>=4.3.0",
        "datasets>=2.12.0",
        "nltk>=3.8.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "pyyaml>=6.0.0",
        "pydantic>=2.0.0"
    ],
    author="NLP team",
    author_email="s-mohamed.eldin@zewailcity.edu.eg",
    description="A hybrid BERT-CNN model for sentiment analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sentiment_hybrid",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)