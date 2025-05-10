import unittest
from src.data_processing import load_tfidf_vectorizer, load_glove_model, load_imdb_dataset
from unittest.mock import patch, mock_open
import os

class TestDataProcessing(unittest.TestCase):
    @patch('src.data_processing.load_dataset')
    def test_load_imdb_dataset(self, mock_load_dataset):
        mock_load_dataset.return_value = {
            'train': {'text': ['test text 1', 'test text 2'], 'label': [0, 1]},
            'test': {'text': ['test text 3'], 'label': [1]}
        }
        train_data, val_data = load_imdb_dataset()
        self.assertEqual(len(train_data['text']), 2)
        self.assertEqual(len(val_data['text']), 1)

    def test_load_tfidf_vectorizer(self):
        mock_train_data = {'text': ['test text 1', 'test text 2']}
        tfidf_vectorizer = load_tfidf_vectorizer(mock_train_data)
        self.assertIsNotNone(tfidf_vectorizer)
        self.assertGreater(len(tfidf_vectorizer.get_feature_names_out()), 0)
        transformed = tfidf_vectorizer.transform(['test text 1']).toarray()
        self.assertEqual(transformed.shape[1], min(5000, len('test text 1 test text 2'.split())))

    @patch('gensim.models.KeyedVectors.load_word2vec_format')
    @patch('builtins.open', new_callable=mock_open, read_data='test 0.1 0.2 0.3\n')
    def test_load_glove_model(self, mock_file, mock_load):
        mock_load.return_value = None 
        glove_model = load_glove_model()
        self.assertIsNotNone(glove_model)
        self.assertTrue(hasattr(glove_model, 'get_vector'))

if __name__ == '__main__':
    unittest.main()