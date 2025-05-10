from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import pickle
import os
import joblib

def load_glove_model(glove_path: str) -> KeyedVectors:
    """Load GloVe word embeddings.

    Args:
        glove_path (str): Path to the GloVe embeddings file.

    Returns:
        KeyedVectors: Loaded GloVe model.

    Raises:
        FileNotFoundError: If the GloVe file does not exist.
    """
    if not os.path.exists(glove_path):
        raise FileNotFoundError(f"GloVe file {glove_path} not found")
    return KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

def load_tfidf_vectorizer(tfidf_path: str) -> TfidfVectorizer:
    """Load a saved TF-IDF vectorizer.

    Args:
        tfidf_path (str): Path to the saved TF-IDF vectorizer.

    Returns:
        TfidfVectorizer: Loaded TF-IDF vectorizer.

    Raises:
        FileNotFoundError: If the TF-IDF file does not exist.
    """
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"TF-IDF vectorizer file {tfidf_path} not found")
    with open(tfidf_path, 'rb') as f:
        return joblib.load(f)