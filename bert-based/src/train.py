import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data_processing import device, tokenizer, tfidf_vectorizer, glove_model, train_loader, val_loader
from src.model import HybridBertCNN

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

def predict_sentiment(model, tokenizer, tfidf_vectorizer, glove_model, text, max_len=128):
    model.eval()
    text = str(text)
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].flatten().to(device)
    attention_mask = encoding['attention_mask'].flatten().to(device)
    tfidf_scores = tfidf_vectorizer.transform([text]).toarray()[0]
    words = text.lower().split()
    tfidf_dict = {word: score for word, score in zip(tfidf_vectorizer.get_feature_names_out(), tfidf_scores) if score > 0}

    glove_embeds = np.zeros((max_len, 300))
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i, token in enumerate(tokens):
        if i >= max_len:
            break
        word = token if not token.startswith('##') and token not in ['[CLS]', '[SEP]', '[PAD]'] else ''
        if word and word in glove_model:
            emb = glove_model[word]
            weight = tfidf_dict.get(word, 1.0) 
            glove_embeds[i] = emb * weight

    glove_embeds = torch.tensor(glove_embeds, dtype=torch.float).to(device)
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), glove_embeds.unsqueeze(0))
        preds = torch.argmax(outputs, dim=1).item()
    label = "Positive" if preds == 1 else "Negative"
    return label, outputs.softmax(dim=1).cpu().numpy()[0]

# Example usage
if __name__ == "__main__":
    from src.data_processing import bert_model
    model = HybridBertCNN(bert_model).to(device)
    train_model(model, train_loader, val_loader)
    example_text = "the person sitting infront of me in the cinema was awful though and kept talking which ruined the experience and made it so bad so I won't be going to this cinema again, but the movie itself is very good"
    predicted_label, probabilities = predict_sentiment(model, tokenizer, tfidf_vectorizer, glove_model, example_text)
    print(f"Predicted Sentiment: {predicted_label}")
    print(f"Probabilities (Negative, Positive): {probabilities}")