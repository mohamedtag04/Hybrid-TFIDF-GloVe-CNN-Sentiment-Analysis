import torch
import numpy as np
from src import device

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