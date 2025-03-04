import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    stop_words = set(stopwords.words("english"))
    
    # Convert to lowercase and remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    
    # Tokenize and filter stop words and short words
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return " ".join(tokens)

def load_and_preprocess_data(train_path, val_path, test_path):
    """Load and preprocess datasets"""
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    # Save original text for analysis
    train_raw = train_data["text"].copy()
    val_raw = val_data["text"].copy()
    test_raw = test_data["text"].copy()
    
    # Apply preprocessing
    train_data["text"] = train_data["text"].apply(preprocess_text)
    val_data["text"] = val_data["text"].apply(preprocess_text)
    test_data["text"] = test_data["text"].apply(preprocess_text)
    
    return train_data, val_data, test_data, train_raw, val_raw, test_raw
