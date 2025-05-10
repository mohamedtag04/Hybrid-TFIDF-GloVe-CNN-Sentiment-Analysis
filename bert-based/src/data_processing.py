import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from datasets import load_dataset
from gensim.models import KeyedVectors
import nltk
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDbDataset(Dataset):
    def __init__(self, data, tokenizer, tfidf_vectorizer, glove_model, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tfidf_vectorizer = tfidf_vectorizer
        self.glove_model = glove_model 
        self.max_len = max_len
        self.texts = data['text']
        self.labels = data['label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        tfidf_scores = self.tfidf_vectorizer.transform([text]).toarray()[0]
        words = text.lower().split()
        tfidf_dict = {word: score for word, score in zip(self.tfidf_vectorizer.get_feature_names_out(), tfidf_scores) if score > 0}

        glove_embeds = np.zeros((self.max_len, 300)) 
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        for i, token in enumerate(tokens):
            if i >= self.max_len:
                break
            word = token if not token.startswith('##') and token not in ['[CLS]', '[SEP]', '[PAD]'] else ''
            if word and word in self.glove_model:
                emb = self.glove_model[word]
                weight = tfidf_dict.get(word, 1.0) 
                glove_embeds[i] = emb * weight

        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'glove_embeds': torch.tensor(glove_embeds, dtype=torch.float).to(device),
            'label': torch.tensor(label, dtype=torch.long).to(device)
        }

# Data loading logic
dataset = load_dataset("imdb")
train_data = dataset['train']
val_data = dataset['test']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectorizer.fit([str(text) for text in train_data['text']])
glove_path = "/kaggle/input/glove6b300dtxt/glove.6B.300d.txt" #obtained from kaggle but needs to be replaced with th local path
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
print("GloVe embeddings loaded successfully.")

train_dataset = IMDbDataset(train_data, tokenizer, tfidf_vectorizer, glove_model)
val_dataset = IMDbDataset(val_data, tokenizer, tfidf_vectorizer, glove_model)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)