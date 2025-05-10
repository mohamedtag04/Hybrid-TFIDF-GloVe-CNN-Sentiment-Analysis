import unittest
import torch
from transformers import BertModel
from src.hybrid_bert_cnn import HybridBertCNN

class TestModel(unittest.TestCase):
    def test_hybrid_bert_cnn_forward(self):
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        model = HybridBertCNN(bert_model)
        input_ids = torch.zeros((1, 128), dtype=torch.int64)
        attention_mask = torch.ones((1, 128), dtype=torch.int64)
        glove_embeds = torch.zeros((1, 128, 300))
        outputs = model(input_ids, attention_mask, glove_embeds)
        self.assertEqual(outputs.size(), torch.Size([1, 2]))

if __name__ == '__main__':
    unittest.main()