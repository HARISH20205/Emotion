import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def analyze_emotion_features(train_data_path="data/train.csv"):
    """Analyze most important features for each emotion class"""
    # Load data
    train_data = pd.read_csv(train_data_path)
    
    # Simple preprocessing
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        return text.lower()
    
    train_data["text"] = train_data["text"].apply(preprocess_text)
    
    # Extract unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)
    X_train = vectorizer.fit_transform(train_data["text"])
    feature_names = vectorizer.get_feature_names_out()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data["label"])
    class_names = label_encoder.classes_
    
    # Get most important features for each class
    class_features = {}
    for class_idx, class_name in enumerate(class_names):
        # Get documents for this class
        class_docs = X_train[y_train == class_idx].toarray()
        
        if class_docs.shape[0] == 0:
            class_features[class_name] = []
            continue
        
        # Sum feature occurrences
        feature_counts = np.sum(class_docs, axis=0)
        
        # Get top features
        top_indices = np.argsort(feature_counts)[-20:][::-1]  # Get top 20 features
        top_features = [(feature_names[idx], feature_counts[idx]) for idx in top_indices]
        class_features[class_name] = top_features
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    for i, (class_name, features) in enumerate(class_features.items()):
        plt.subplot(3, 2, i+1)
        
        if not features:
            plt.text(0.5, 0.5, f"No data for class: {class_name}", 
                     horizontalalignment='center', verticalalignment='center')
            continue
            
        x = [f[0] for f in features]
        y = [f[1] for f in features]
        
        plt.barh(x, y)
        plt.title(f"Top Features for '{class_name}'")
        plt.xlabel("Frequency")
        plt.tight_layout()
    
    plt.savefig('emotion_features.png')
    print("Feature analysis chart saved as 'emotion_features.png'")
    
    # Count class distribution
    class_counts = Counter(train_data["label"])
    
    # Create class distribution chart
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Emotion Class Distribution in Training Data")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print("Class distribution chart saved as 'class_distribution.png'")
    
    return class_features

if __name__ == "__main__":
    analyze_emotion_features()
