import unittest
import torch
import numpy as np
from unittest.mock import MagicMock
from src.imdb_dataset import IMDbDataset
from src import device
from transformers import T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

class TestIMDbDataset(unittest.TestCase):
    def setUp(self):
        # Mock data
        self.mock_data = {
            'text': ["I love this movie!", "This film is terrible."],
            'label': [1, 0]
        }
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
        
        # Mock TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_vectorizer.fit(self.mock_data['text'])
        
        # Mock GloVe model
        self.glove_model = MagicMock(spec=KeyedVectors)
        self.glove_model.__contains__.side_effect = lambda word: word in ["love", "movie", "film", "terrible", "this", "is"]
        self.glove_model.__getitem__.side_effect = lambda word: np.ones(300) if word in ["love", "movie", "film", "terrible", "this", "is"] else np.zeros(300)
        
        # Initialize dataset
        self.dataset = IMDbDataset(self.mock_data, self.tokenizer, self.tfidf_vectorizer, self.glove_model, max_len=10)

    def test_dataset_length(self):
        # Test if the dataset length matches the mock data
        self.assertEqual(len(self.dataset), 2)

    def test_getitem_structure(self):
        # Test the structure of the returned item
        item = self.dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('glove_embeds', item)
        self.assertIn('label', item)

        # Check shapes and types
        self.assertEqual(item['input_ids'].shape, torch.Size([10]))
        self.assertEqual(item['attention_mask'].shape, torch.Size([10]))
        self.assertEqual(item['glove_embeds'].shape, torch.Size([10, 300]))
        self.assertEqual(item['label'].shape, torch.Size([]))
        self.assertTrue(isinstance(item['label'], torch.Tensor))

    def test_getitem_content(self):
        # Test the content of the first item (positive example)
        item = self.dataset[0]
        self.assertEqual(item['label'].item(), 1)  # Positive label
        self.assertTrue((item['attention_mask'] == 1).sum() > 0)  # At least some tokens are not padded

        # Test GloVe embeddings weighting by TF-IDF
        glove_embeds = item['glove_embeds'].cpu().numpy()
        self.assertTrue(np.any(glove_embeds != 0))  # Some embeddings should be non-zero due to GloVe and TF-IDF
        tfidf_scores = self.tfidf_vectorizer.transform(["I love this movie!"]).toarray()[0]
        self.assertTrue(np.all(glove_embeds[:len("I love this movie!".split())] == np.ones(300) * tfidf_scores[:len("I love this movie!".split())]))

    def test_getitem_negative_example(self):
        # Test the content of the second item (negative example)
        item = self.dataset[1]
        self.assertEqual(item['label'].item(), 0)  # Negative label
        self.assertTrue((item['attention_mask'] == 1).sum() > 0)  # At least some tokens are not padded

        # Test GloVe embeddings weighting by TF-IDF
        glove_embeds = item['glove_embeds'].cpu().numpy()
        self.assertTrue(np.any(glove_embeds != 0))  # Some embeddings should be non-zero due to GloVe and TF-IDF
        tfidf_scores = self.tfidf_vectorizer.transform(["This film is terrible."]).toarray()[0]
        self.assertTrue(np.all(glove_embeds[:len("This film is terrible.".split())] == np.ones(300) * tfidf_scores[:len("This film is terrible.".split())]))

if __name__ == '__main__':
    unittest.main()