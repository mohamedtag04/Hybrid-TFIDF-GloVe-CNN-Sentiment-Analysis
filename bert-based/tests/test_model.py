import unittest
import torch
from src.model import HybridBertCNN
from src.data_processing import device
from transformers import BertModel

class TestModel(unittest.TestCase):
    def setUp(self):
        # Initialize BERT model
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        
        # Initialize the HybridBertCNN model
        self.model = HybridBertCNN(self.bert_model, glove_dim=300, dropout=0.3).to(device)
        
        # Mock inputs (simulating IMDbDataset output)
        self.batch_size = 2
        self.max_len = 10
        self.input_ids = torch.ones((self.batch_size, self.max_len), dtype=torch.long).to(device)
        self.attention_mask = torch.ones((self.batch_size, self.max_len), dtype=torch.long).to(device)
        self.glove_embeds = torch.ones((self.batch_size, self.max_len, 300), dtype=torch.float).to(device)

    def test_forward_pass(self):
        # Test the forward pass of the model
        output = self.model(self.input_ids, self.attention_mask, self.glove_embeds)
        
        # Check output shape (batch_size, num_classes)
        self.assertEqual(output.shape, torch.Size([self.batch_size, 2]))

    def test_output_range(self):
        # Test if the output logits are reasonable (not NaN or infinite)
        output = self.model(self.input_ids, self.attention_mask, self.glove_embeds)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_model_with_zero_attention_mask(self):
        # Test the model with an attention mask of all zeros
        attention_mask_zeros = torch.zeros_like(self.attention_mask)
        output = self.model(self.input_ids, attention_mask_zeros, self.glove_embeds)
        self.assertEqual(output.shape, torch.Size([self.batch_size, 2]))
        self.assertFalse(torch.isnan(output).any())

if __name__ == '__main__':
    unittest.main()