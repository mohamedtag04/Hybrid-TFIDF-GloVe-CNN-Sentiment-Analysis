import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from datasets import load_dataset
from gensim.models import KeyedVectors
from tqdm.auto import tqdm
import nltk

# Download NLTK data
nltk.download('punkt')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")