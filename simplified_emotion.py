import os
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Load data
train_data = pd.read_csv("data/train.csv")
val_data = pd.read_csv("data/val.csv")
test_data = pd.read_csv("data/test.csv")

# Save raw text for sentiment analysis
train_raw = train_data["text"].copy()
val_raw = val_data["text"].copy()
test_raw = test_data["text"].copy()

def preprocess_text(text):
    """Simple preprocessing function focusing only on the most effective steps"""
    if not isinstance(text, str):
        return ""
        
    stop_words = set(stopwords.words("english"))
    
    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    
    # Split into tokens, filter stop words and short words
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return " ".join(tokens)

# Preprocess text
train_data["text"] = train_data["text"].apply(preprocess_text)
val_data["text"] = val_data["text"].apply(preprocess_text)
test_data["text"] = test_data["text"].apply(preprocess_text)

# Create unigram and bigram features with simplified parameters
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
X_train = vectorizer.fit_transform(train_data["text"])
X_val = vectorizer.transform(val_data["text"])
X_test = vectorizer.transform(test_data["text"])

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["label"])
y_val = label_encoder.transform(val_data["label"])
y_test = label_encoder.transform(test_data["label"])

# Calculate term frequencies for each emotion class
class_term_frequencies = {}
for emotion in range(len(label_encoder.classes_)):
    emotion_docs = X_train[y_train == emotion]
    if emotion_docs.shape[0] > 0:
        # Average term frequency for this emotion
        class_term_frequencies[emotion] = np.sum(emotion_docs, axis=0).A.flatten() / emotion_docs.shape[0]
    else:
        class_term_frequencies[emotion] = np.zeros(X_train.shape[1])

# Create a simplified fuzzy logic system
# Define fuzzy variables
term_similarity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'term_similarity')
term_intensity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'term_intensity')
emotion_class = ctrl.Consequent(np.arange(0, len(label_encoder.classes_)), 'emotion_class')

# Define membership functions - simplified to just 3 levels for each input
term_similarity['low'] = fuzz.trimf(term_similarity.universe, [0, 0, 0.4])
term_similarity['medium'] = fuzz.trimf(term_similarity.universe, [0.3, 0.5, 0.7])
term_similarity['high'] = fuzz.trimf(term_similarity.universe, [0.6, 1, 1])

term_intensity['low'] = fuzz.trimf(term_intensity.universe, [0, 0, 0.4])
term_intensity['medium'] = fuzz.trimf(term_intensity.universe, [0.3, 0.5, 0.7])
term_intensity['high'] = fuzz.trimf(term_intensity.universe, [0.6, 1, 1])

# Define output membership functions - one for each emotion class
emotions = label_encoder.classes_
for i, emotion in enumerate(emotions):
    # Create triangular membership functions centered around each class index
    lower = max(0, i-0.5)
    upper = min(len(emotions)-1, i+0.5)
    emotion_class[emotion] = fuzz.trimf(emotion_class.universe, [lower, i, upper])

# Create simplified rules
rules = []

# Map emotion names to common characteristics
emotion_characteristics = {
    'sadness': {'similarity': 'high', 'intensity': 'medium'},
    'joy': {'similarity': 'high', 'intensity': 'high'},
    'love': {'similarity': 'medium', 'intensity': 'high'},
    'anger': {'similarity': 'high', 'intensity': 'high'},
    'fear': {'similarity': 'medium', 'intensity': 'medium'},
    'surprise': {'similarity': 'low', 'intensity': 'medium'}
}

# Create rules based on emotion characteristics
for i, emotion in enumerate(emotions):
    if emotion in emotion_characteristics:
        char = emotion_characteristics[emotion]
        rules.append(ctrl.Rule(
            term_similarity[char['similarity']] & 
            term_intensity[char['intensity']], 
            emotion_class[emotion]
        ))
    else:
        # Default rule if emotion is not in our mapping
        rules.append(ctrl.Rule(
            term_similarity['medium'] & 
            term_intensity['medium'], 
            emotion_class[emotion]
        ))

# Create control system
emotion_ctrl = ctrl.ControlSystem(rules)
emotion_sim = ctrl.ControlSystemSimulation(emotion_ctrl)

def predict_emotion(sample_vector, raw_text=None):
    """Predict emotion using simplified fuzzy logic"""
    sample = sample_vector.toarray().flatten()
    
    # Calculate term similarity and intensity scores for each emotion class
    similarity_scores = []
    intensity_scores = []
    
    for emotion in range(len(emotions)):
        # Cosine similarity between sample and emotion term frequencies
        emotion_freq = class_term_frequencies[emotion]
        dot_product = np.dot(sample, emotion_freq)
        magnitude1 = np.sqrt(np.sum(sample**2))
        magnitude2 = np.sqrt(np.sum(emotion_freq**2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            similarity = 0
        else:
            similarity = dot_product / (magnitude1 * magnitude2)
        
        similarity_scores.append(similarity)
        
        # Term intensity: count of non-zero terms relative to vocabulary size
        intensity = np.count_nonzero(sample) / max(len(sample), 1)
        intensity_scores.append(intensity)
    
    # Use fuzzy logic to determine emotion
    try:
        emotion_sim.input['term_similarity'] = max(similarity_scores)
        emotion_sim.input['term_intensity'] = min(1.0, sum(intensity_scores) / max(len(intensity_scores), 1))
        emotion_sim.compute()
        
        # Get the defuzzified output
        emotion_idx = round(emotion_sim.output['emotion_class'])
        
        # Ensure prediction is within valid range
        return max(0, min(len(emotions)-1, emotion_idx))
    except:
        # Fallback to highest similarity if fuzzy system fails
        return np.argmax(similarity_scores)

def predict_batch(X, raw_texts=None):
    """Predict emotions for a batch of texts"""
    predictions = []
    
    for i in range(X.shape[0]):
        predictions.append(predict_emotion(X[i], None if raw_texts is None else raw_texts[i]))
        
    return predictions

# Make predictions
print("Making predictions...")
y_val_pred = predict_batch(X_val, val_raw)
y_test_pred = predict_batch(X_test, test_raw)

# Evaluate model performance
print("\nValidation Results:")
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Accuracy: {val_accuracy:.4f}")
print(classification_report(y_val, y_val_pred, target_names=emotions))

print("\nTest Results:")
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, y_test_pred, target_names=emotions))

# Plot confusion matrix (if matplotlib is available)
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotions)
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap='Blues')
    plt.title('Emotion Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
except ImportError:
    print("\nMatplotlib not available. Skipping confusion matrix visualization.")

print("\nDone!")
