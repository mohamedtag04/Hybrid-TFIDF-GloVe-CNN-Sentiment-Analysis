# Sentiment Hybrid

A Python package for sentiment analysis using a hybrid BERT-CNN model combined with GloVe embeddings and TF-IDF weighting.

## Installation

1. Clone the repository:

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package:
   ```bash
   pip install .
   ```

4. Ensure the following files are in place:
   - `weights/hybrid_bert_cnn.pth`: Trained model weights.
   - `weights/tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer.
   - `data/glove/glove.6B.300d.txt`: GloVe embeddings.

## Usage

### Running the API

Start the FastAPI server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Access the API at `http://localhost:8000/docs` for interactive documentation.

Example API call using `curl`:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "This movie was great!"}'
```

### Using the Predictor Locally

```python
from sentiment_hybrid.predict import SentimentPredictor

predictor = SentimentPredictor()
label, probs = predictor.predict_sentiment("This movie was great!")
print(f"Sentiment: {label}, Probabilities: {probs}")
```


## Directory Structure

```
sentiment_hybrid/
├── sentiment_hybrid/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── predict.py
│   └── utils.py
├── api/
│   ├── __init__.py
│   └── main.py
├── weights/
│   ├── hybrid_bert_cnn.pth
│   └── tfidf_vectorizer.pkl
├── data/
│   └── glove/
│       └── glove.6B.300d.txt
├── config.yaml
├── setup.py
├── README.md
└── requirements.txt
```

## Requirements

See `requirements.txt` for a full list of dependencies.

## License

MIT License