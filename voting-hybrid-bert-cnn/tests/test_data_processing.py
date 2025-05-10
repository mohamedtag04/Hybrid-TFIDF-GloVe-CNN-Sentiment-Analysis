import unittest
from src.data_processing import load_imdb_dataset, prepare_tfidf, load_glove_embeddings

class TestDataProcessing(unittest.TestCase):
    def test_load_imdb_dataset(self):
        dataset = load_imdb_dataset()
        self.assertIn('train', dataset)
        self.assertIn('test', dataset)

    def test_prepare_tfidf(self):
        texts = ['sample sentence', 'another sample sentence']
        vectorizer = prepare_tfidf(texts)
        features = vectorizer.transform(['sample'])
        self.assertEqual(features.shape[1], 5000)

    def test_load_glove_embeddings(self):
        glove_path = '/path/to/glove.6B.300d.txt'
        glove_model = load_glove_embeddings(glove_path)
        self.assertIn('word', glove_model)

if __name__ == '__main__':
    unittest.main()