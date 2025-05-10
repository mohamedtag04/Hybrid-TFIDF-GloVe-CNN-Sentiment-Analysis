from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

def load_imdb_dataset():
    return load_dataset("imdb")

def prepare_tfidf(train_texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit([str(text) for text in train_texts])
    return vectorizer

def load_glove_embeddings(glove_path):
    return KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)