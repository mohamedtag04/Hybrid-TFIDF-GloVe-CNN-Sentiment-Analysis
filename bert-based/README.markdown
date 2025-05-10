# Sentiment Analysis with Hybrid BERT-CNN Model

This project implements a hybrid BERT-CNN model for sentiment analysis on the IMDb dataset. It combines BERT embeddings with GloVe embeddings weighted by TF-IDF scores, processes them through a CNN, and uses attention mechanisms for classification.
## Setup

1. Install dependencies: `pip install -r requirements.txt`.
2. Ensure GloVe embeddings are available at the specified path.
3. Run the training script: `python src/train.py`.

## Usage
The model can be trained and used for sentiment prediction as shown in `train.py`.
