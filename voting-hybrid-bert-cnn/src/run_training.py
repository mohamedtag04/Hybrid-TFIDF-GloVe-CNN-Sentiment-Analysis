import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from src.data_processing import load_imdb_dataset, prepare_tfidf, load_glove_embeddings
from src.imdb_dataset import IMDbDataset
from src.hybrid_bert_cnn import HybridBertCNN
from src.train import train_model, save_model

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare dataset
    dataset = load_imdb_dataset()
    train_data = dataset['train']
    val_data = dataset['test']

    # Tokenizer and BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # TF-IDF Vectorizer preparation
    tfidf_vectorizer = prepare_tfidf(train_data['text'])

    # Load GloVe model
    glove_path = "./src/glove/glove.6B.300d.txt" 
    glove_model = load_glove_embeddings(glove_path)
    
    # DataLoader setup
    train_dataset = IMDbDataset(train_data, tokenizer, tfidf_vectorizer, glove_model)
    val_dataset = IMDbDataset(val_data, tokenizer, tfidf_vectorizer, glove_model)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model initialization
    model = HybridBertCNN(bert_model).to(device)

    # Training
    train_model(model, train_loader, val_loader, epochs=5, lr=2e-5)

    # Save the trained model
    save_model(model, "./src/output/hybrid_bert_cnn_model.pth")

if __name__ == "__main__":
    main()