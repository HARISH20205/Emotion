"""
Simple runner script to compare different emotion classification approaches
"""
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score

# Ensure data directory exists
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")
    print("Please place your train.csv, val.csv, and test.csv files in the data directory.")
    exit()

# Function to run simplified fuzzy logic approach
def run_fuzzy_approach():
    print("\n=== Running Simplified Fuzzy Logic Approach ===")
    start_time = time.time()
    
    # Import here to avoid circular imports
    from simplified_emotion import predict_batch
    
    # Load and vectorize data (similar code exists in simplified_emotion.py)
    import re
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
            
        stop_words = set(stopwords.words("english"))
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(tokens)
    
    # Load data
    train_data = pd.read_csv("data/train.csv")
    val_data = pd.read_csv("data/val.csv")
    test_data = pd.read_csv("data/test.csv")
    
    # Preprocess
    train_data["text"] = train_data["text"].apply(preprocess_text)
    val_data["text"] = val_data["text"].apply(preprocess_text)
    test_data["text"] = test_data["text"].apply(preprocess_text)
    
    # Vectorize
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
    X_train = vectorizer.fit_transform(train_data["text"])
    X_test = vectorizer.transform(test_data["text"])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data["label"])
    y_test = label_encoder.transform(test_data["label"])
    
    # Make predictions
    test_pred = predict_batch(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, test_pred)
    
    end_time = time.time()
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    
    return accuracy

# Function to run hybrid approach
def run_hybrid_approach():
    print("\n=== Running Hybrid Approach ===")
    start_time = time.time()
    
    # Import here to avoid circular imports
    from hybrid_model import HybridEmotionClassifier
    
    # Load data
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    
    # Train model
    model = HybridEmotionClassifier()
    model.fit(train_data["text"], train_data["label"])
    
    # Make predictions
    test_pred = model.predict(test_data["text"])
    test_true = model.label_encoder.transform(test_data["label"])
    
    # Calculate accuracy
    accuracy = accuracy_score(test_true, test_pred)
    
    end_time = time.time()
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    
    return accuracy

if __name__ == "__main__":
    print("=== Emotion Classification Comparison ===")
    
    # Check if data files exist
    data_files = ["train.csv", "val.csv", "test.csv"]
    missing_files = [f for f in data_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing_files:
        print(f"Error: Missing data files: {', '.join(missing_files)}")
        print(f"Please place the missing files in the {data_dir}/ directory.")
        exit()
    
    # Run approaches
    try:
        fuzzy_accuracy = run_fuzzy_approach()
    except Exception as e:
        print(f"Error running fuzzy approach: {str(e)}")
        fuzzy_accuracy = 0
    
    try:
        hybrid_accuracy = run_hybrid_approach()
    except Exception as e:
        print(f"Error running hybrid approach: {str(e)}")
        hybrid_accuracy = 0
    
    # Print comparison
    print("\n=== Results Comparison ===")
    print(f"Simplified Fuzzy Logic: {fuzzy_accuracy:.4f}")
    print(f"Hybrid Approach: {hybrid_accuracy:.4f}")
    
    # Analyze features
    try:
        from feature_analysis import analyze_emotion_features
        print("\nAnalyzing emotion features...")
        analyze_emotion_features()
    except Exception as e:
        print(f"Error analyzing features: {str(e)}")
