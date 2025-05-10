from typing import Tuple, Optional
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import numpy as np
from .model import HybridBertCNN
from .config import CONFIG
from .utils import load_glove_model, load_tfidf_vectorizer

class SentimentPredictor:
    """Class for predicting sentiment using the HybridBertCNN model.

    Args:
        model_path (str, optional): Path to the trained model weights.
        tfidf_path (str, optional): Path to the saved TF-IDF vectorizer.
        glove_path (str, optional): Path to the GloVe embeddings.
        max_len (int, optional): Maximum sequence length. Defaults to 128.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        tfidf_path: Optional[str] = None,
        glove_path: Optional[str] = None,
        max_len: int = 128
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.model = HybridBertCNN(self.bert_model).to(self.device)
        self.max_len = max_len

        # Load model weights
        model_path = model_path or CONFIG['weights']['model']
        if not torch.cuda.is_available():
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load TF-IDF vectorizer
        tfidf_path = tfidf_path or CONFIG['weights']['tfidf']
        self.tfidf_vectorizer = load_tfidf_vectorizer(tfidf_path)

        # Load GloVe model
        glove_path = glove_path or CONFIG['data']['glove']
        self.glove_model = load_glove_model(glove_path)

    def predict_sentiment(self, text: str) -> Tuple[str, np.ndarray]:
        """Predict sentiment for a given text.

        Args:
            text (str): Input text for sentiment analysis.

        Returns:
            Tuple[str, np.ndarray]: Predicted sentiment label ("Positive" or "Negative")
                                   and probabilities for each class.
        """
        text = str(text)
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

        input_ids = encoding['input_ids'].flatten().to(self.device)
        attention_mask = encoding['attention_mask'].flatten().to(self.device)
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

        glove_embeds = torch.tensor(glove_embeds, dtype=torch.float).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), glove_embeds.unsqueeze(0))
            preds = torch.argmax(outputs, dim=1).item()
        label = "Positive" if preds == 1 else "Negative"
        probabilities = outputs.softmax(dim=1).cpu().numpy()[0]
        return label, probabilities