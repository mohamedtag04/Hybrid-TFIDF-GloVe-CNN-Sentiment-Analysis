from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from src import device
from src.data_processing import load_tfidf_vectorizer, load_glove_model, load_imdb_dataset
from src.imdb_dataset import IMDbDataset
from src.hybrid_bert_cnn import HybridT5CNN
from src.model import train_model

def main():
    # Load dataset
    train_data, val_data = load_imdb_dataset()

    # Load tokenizer and T5 encoder
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
    t5_encoder = T5EncoderModel.from_pretrained("google-t5/t5-base").to(device)

    # Load TF-IDF and GloVe
    tfidf_vectorizer = load_tfidf_vectorizer(train_data)
    glove_model = load_glove_model()

    # Create datasets and dataloaders
    train_dataset = IMDbDataset(train_data, tokenizer, tfidf_vectorizer, glove_model)
    val_dataset = IMDbDataset(val_data, tokenizer, tfidf_vectorizer, glove_model)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize and train model
    model = HybridT5CNN(t5_encoder).to(device)
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()