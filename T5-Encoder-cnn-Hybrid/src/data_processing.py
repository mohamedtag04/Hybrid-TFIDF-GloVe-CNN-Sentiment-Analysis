from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from datasets import load_dataset

def load_tfidf_vectorizer(train_data, max_features=5000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_vectorizer.fit([str(text) for text in train_data['text']])
    return tfidf_vectorizer

def load_glove_model(glove_path="/kaggle/input/glove6b300dtxt/glove.6B.300d.txt"):
    glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
    print("GloVe embeddings loaded successfully.")
    return glove_model

def load_imdb_dataset():
    dataset = load_dataset("imdb")
    train_data = dataset['train']
    val_data = dataset['test']
    return train_data, val_data