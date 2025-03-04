import os
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datasets import load_dataset

nltk.download("stopwords")
from nltk.corpus import stopwords

train_data = pd.read_csv("data/train.csv")
val_data = pd.read_csv("data/val.csv")
test_data = pd.read_csv("data/test.csv")

# Enhanced preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]  # Filter short words
    return " ".join(tokens)

train_data["text"] = train_data["text"].apply(preprocess_text)
val_data["text"] = val_data["text"].apply(preprocess_text)
test_data["text"] = test_data["text"].apply(preprocess_text)

# Apply unigram and bigram embedding with improved parameters
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
X_train = vectorizer.fit_transform(train_data["text"])
X_val = vectorizer.transform(val_data["text"])
X_test = vectorizer.transform(test_data["text"])

# Get emotion class distribution
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["label"])
y_val = label_encoder.transform(val_data["label"])
y_test = label_encoder.transform(test_data["label"])

# Get vocabulary size for normalization
vocab_size = len(vectorizer.vocabulary_)

# Calculate emotion-specific term frequencies
emotion_term_freqs = {}
for emotion in range(6):  # Assuming 6 emotion classes
    emotion_docs = X_train[y_train == emotion]
    if emotion_docs.shape[0] > 0:
        emotion_term_freqs[emotion] = np.sum(emotion_docs, axis=0).A.flatten() / emotion_docs.shape[0]
    else:
        emotion_term_freqs[emotion] = np.zeros(X_train.shape[1])

# Improved fuzzy logic system
# Define input variables
feature_density = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'feature_density')
emotion_signal = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'emotion_signal')
label_match = ctrl.Consequent(np.arange(0, 6), 'label_match')

# Define membership functions for inputs
feature_density['low'] = fuzz.trimf(feature_density.universe, [0, 0, 0.4])
feature_density['medium'] = fuzz.trimf(feature_density.universe, [0.2, 0.5, 0.8])
feature_density['high'] = fuzz.trimf(feature_density.universe, [0.6, 1, 1])

emotion_signal['weak'] = fuzz.trimf(emotion_signal.universe, [0, 0, 0.4])
emotion_signal['moderate'] = fuzz.trimf(emotion_signal.universe, [0.3, 0.5, 0.7])
emotion_signal['strong'] = fuzz.trimf(emotion_signal.universe, [0.6, 1, 1])

# Define membership functions for output - balanced for each emotion
label_match['sadness'] = fuzz.trimf(label_match.universe, [0, 0, 1])
label_match['joy'] = fuzz.trimf(label_match.universe, [0.8, 1.1, 1.9])
label_match['love'] = fuzz.trimf(label_match.universe, [1.8, 2.1, 2.9])
label_match['anger'] = fuzz.trimf(label_match.universe, [2.8, 3.1, 3.9])
label_match['fear'] = fuzz.trimf(label_match.universe, [3.8, 4.1, 4.9])
label_match['surprise'] = fuzz.trimf(label_match.universe, [4.8, 5, 5])

# Define more balanced fuzzy rules
rules = [
    ctrl.Rule(feature_density['medium'] & emotion_signal['strong'], label_match['sadness']),
    ctrl.Rule(feature_density['high'] & emotion_signal['strong'], label_match['joy']),
    ctrl.Rule(feature_density['medium'] & emotion_signal['moderate'], label_match['love']),
    ctrl.Rule(feature_density['high'] & emotion_signal['weak'], label_match['anger']),
    ctrl.Rule(feature_density['low'] & emotion_signal['moderate'], label_match['fear']),
    ctrl.Rule(feature_density['medium'] & emotion_signal['weak'], label_match['surprise']),
]

# Create control system
emotion_ctrl = ctrl.ControlSystem(rules)
emotion_sim = ctrl.ControlSystemSimulation(emotion_ctrl)

# Improved prediction function
def predict_emotions(X):
    predictions = []
    
    for i in range(X.shape[0]):
        sample = X[i].toarray().flatten()
        
        # Calculate feature density
        non_zero_features = np.count_nonzero(sample)
        density_score = min(non_zero_features / vocab_size, 1.0)
        
        # Calculate emotion signals for each class
        emotion_scores = []
        for emotion in range(6):
            # Dot product of sample vector with emotion term frequency
            emotion_score = np.dot(sample, emotion_term_freqs[emotion]) / max(np.sum(sample), 1)
            emotion_scores.append(emotion_score)
        
        # Get the highest emotion signal
        max_emotion_signal = max(emotion_scores) if emotion_scores else 0
        
        try:
            # Use fuzzy logic to predict
            emotion_sim.input['feature_density'] = density_score
            emotion_sim.input['emotion_signal'] = min(max_emotion_signal, 1.0)
            emotion_sim.compute()
            emotion_index = round(emotion_sim.output['label_match'])
            
            # Ensure the emotion index is valid
            emotions_range = list(range(6))
            if emotion_index not in emotions_range:
                emotion_index = min(emotions_range, key=lambda x: abs(x - emotion_index))
            
            predictions.append(emotion_index)
        except:
            # Fallback to the emotion with highest signal if fuzzy logic fails
            predictions.append(np.argmax(emotion_scores))
    
    return predictions

# Define the label mapping
label_mapping = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
target_names = [label_mapping[i] for i in range(len(label_mapping))]

# Predict emotions
y_val_pred = predict_emotions(X_val)
y_test_pred = predict_emotions(X_test)

# Evaluate the model
print("Validation Metrics:")
print(classification_report(y_val, y_val_pred, target_names=target_names))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

print("\nTest Metrics:")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
