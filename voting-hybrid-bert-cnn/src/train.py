import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, val_loader, epochs=3, lr=2e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            glove_embeds = batch['glove_embeds']
            labels = batch['label']

            outputs = model(input_ids, attention_mask, glove_embeds)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                glove_embeds = batch['glove_embeds']
                labels = batch['label']

                outputs = model(input_ids, attention_mask, glove_embeds)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Val Accuracy: {correct / total:.4f}")

def save_model(model, filepath="hybrid_bert_cnn_model.pth"):
    torch.save(model.state_dict(), filepath)

def load_model(filepath, model_class):
    model = model_class()
    model.load_state_dict(torch.load(filepath))
    return model