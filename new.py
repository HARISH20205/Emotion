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
import pickle

# nltk.download("stopwords")
from nltk.corpus import stopwords

# Load dataset
train_data = pd.read_csv("data/train.csv")
val_data = pd.read_csv("data/val.csv")
test_data = pd.read_csv("data/test.csv")


# Enhanced preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(tokens)


train_data["text"] = train_data["text"].apply(preprocess_text)
val_data["text"] = val_data["text"].apply(preprocess_text)
test_data["text"] = test_data["text"].apply(preprocess_text)

vectorizer = CountVectorizer(
    ngram_range=(1, 4),
    min_df=2,
    max_df=0.95,
    max_features=5000,
)
X_train = vectorizer.fit_transform(train_data["text"])
X_val = vectorizer.transform(val_data["text"])
X_test = vectorizer.transform(test_data["text"])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["label"])
y_val = label_encoder.transform(val_data["label"])
y_test = label_encoder.transform(test_data["label"])

vocab_size = len(vectorizer.vocabulary_)


def calculate_emotion_weights(X_train, y_train, num_classes):
    weights = np.zeros((num_classes, X_train.shape[1]))
    for emotion in range(num_classes):
        emotion_docs = X_train[y_train == emotion]
        if emotion_docs.shape[0] > 0:
            term_freq = np.sum(emotion_docs, axis=0).A.flatten()
            doc_freq = np.sum(emotion_docs > 0, axis=0).A.flatten()
            weights[emotion] = (term_freq / emotion_docs.shape[0]) * np.log(
                emotion_docs.shape[0] / (doc_freq + 1)
            )
    return weights


num_classes = len(label_encoder.classes_)
emotion_weights = calculate_emotion_weights(X_train, y_train, num_classes)

feature_density = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "feature_density")
emotion_signal = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "emotion_signal")
label_match = ctrl.Consequent(np.arange(0, num_classes), "label_match")

feature_density["low"] = fuzz.trimf(feature_density.universe, [0, 0, 0.3])
feature_density["medium"] = fuzz.trimf(feature_density.universe, [0.2, 0.4, 0.7])
feature_density["high"] = fuzz.trimf(feature_density.universe, [0.5, 0.8, 1])

emotion_signal["weak"] = fuzz.trimf(emotion_signal.universe, [0, 0, 0.3])
emotion_signal["moderate"] = fuzz.trimf(emotion_signal.universe, [0.2, 0.5, 0.8])
emotion_signal["strong"] = fuzz.trimf(emotion_signal.universe, [0.6, 1, 1])

for i in range(num_classes):
    label_match[label_encoder.inverse_transform([i])[0]] = fuzz.trimf(
        label_match.universe, [i - 0.2, i, i + 0.2]
    )

rules = []
for i in range(num_classes):
    rules.append(
        ctrl.Rule(
            feature_density["high"] & emotion_signal["strong"],
            label_match[label_encoder.inverse_transform([i])[0]],
        )
    )

emotion_ctrl = ctrl.ControlSystem(rules)
emotion_sim = ctrl.ControlSystemSimulation(emotion_ctrl)


def predict_emotions(X, y_train, num_classes, vocab_size, emotion_weights):
    predictions = []
    for i in range(X.shape[0]):
        sample = X[i].toarray().flatten()
        density_score = min(np.count_nonzero(sample) / vocab_size, 1.0)
        emotion_scores = []
        for emotion in range(num_classes):
            weighted_score = np.dot(sample, emotion_weights[emotion])
            normalized_score = weighted_score / (np.sum(sample) + 1)
            emotion_scores.append(normalized_score)

        max_emotion_signal = max(emotion_scores) if emotion_scores else 0

        try:
            emotion_sim.input["feature_density"] = density_score
            emotion_sim.input["emotion_signal"] = min(max_emotion_signal, 1.0)
            emotion_sim.compute()
            emotion_index = round(emotion_sim.output["label_match"])
            emotions_range = list(range(num_classes))
            if emotion_index not in emotions_range:
                emotion_index = min(
                    emotions_range, key=lambda x: abs(x - emotion_index)
                )
            predictions.append(emotion_index)
        except:
            predictions.append(np.argmax(emotion_scores))
    return predictions


for epoch in range(10):
    y_train_pred = predict_emotions(
        X_train, y_train, num_classes, vocab_size, emotion_weights
    )
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Epoch {epoch + 1}/10")
    print(f"Train Accuracy: {train_accuracy:.2f}")

    # Update fuzzy values
    emotion_weights = calculate_emotion_weights(X_train, y_train, num_classes)

label_mapping = {i: label_encoder.inverse_transform([i])[0] for i in range(num_classes)}
target_names = [label_mapping[i] for i in range(num_classes)]

y_val_pred = predict_emotions(X_val, y_train, num_classes, vocab_size, emotion_weights)
y_test_pred = predict_emotions(
    X_test, y_train, num_classes, vocab_size, emotion_weights
)

print("Validation Metrics:")
try:
    print(classification_report(y_val, y_val_pred, target_names=target_names))
except Exception as e:
    print("Error in classification report for validation:", e)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

print("\nTest Metrics:")
try:
    print(classification_report(y_test, y_test_pred, target_names=target_names))
except Exception as e:
    print("Error in classification report for test:", e)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
