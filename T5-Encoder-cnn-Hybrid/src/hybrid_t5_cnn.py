import torch
import torch.nn as nn
from src import device

class HybridT5CNN(nn.Module):
    def __init__(self, t5_encoder, glove_dim=300, dropout=0.3):
        super(HybridT5CNN, self).__init__()
        self.t5 = t5_encoder
        t5_dim = 768
        self.conv3 = nn.Conv1d(in_channels=t5_dim + glove_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=t5_dim + glove_dim, out_channels=128, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(in_channels=t5_dim + glove_dim, out_channels=128, kernel_size=5, padding=2)
        self.attention = nn.MultiheadAttention(embed_dim=384, num_heads=8)
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, glove_embeds):
        with torch.no_grad():
            t5_output = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        t5_embeds = t5_output.last_hidden_state
        combined_embeds = torch.cat((t5_embeds, glove_embeds), dim=-1)
        combined_embeds = combined_embeds.permute(0, 2, 1)
        conv3_out = self.relu(self.conv3(combined_embeds))
        conv4_out = self.relu(self.conv4(combined_embeds))
        conv5_out = self.relu(self.conv5(combined_embeds))
        conv3_out = torch.max(conv3_out, dim=-1)[0]
        conv4_out = torch.max(conv4_out, dim=-1)[0]
        conv5_out = torch.max(conv5_out, dim=-1)[0]
        cnn_out = torch.cat((conv3_out, conv4_out, conv5_out), dim=-1)
        cnn_out = cnn_out.unsqueeze(0)
        attn_out, _ = self.attention(cnn_out, cnn_out, cnn_out)
        attn_out = attn_out.squeeze(0)
        out = self.dropout(attn_out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out