"""
A hybrid approach using both fuzzy logic and traditional machine learning
to achieve better emotion classification results.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import re
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

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

class HybridEmotionClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
        self.label_encoder = LabelEncoder()
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.class_term_frequencies = {}
        self.fuzzy_system = None
        
    def _create_fuzzy_system(self, num_classes):
        # Define fuzzy variables
        ml_confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'ml_confidence')
        term_similarity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'term_similarity')
        emotion_class = ctrl.Consequent(np.arange(0, num_classes), 'emotion_class')
        
        # Define membership functions
        ml_confidence['low'] = fuzz.trimf(ml_confidence.universe, [0, 0, 0.4])
        ml_confidence['medium'] = fuzz.trimf(ml_confidence.universe, [0.3, 0.5, 0.7])
        ml_confidence['high'] = fuzz.trimf(ml_confidence.universe, [0.6, 1, 1])
        
        term_similarity['low'] = fuzz.trimf(term_similarity.universe, [0, 0, 0.4])
        term_similarity['medium'] = fuzz.trimf(term_similarity.universe, [0.3, 0.5, 0.7])
        term_similarity['high'] = fuzz.trimf(term_similarity.universe, [0.6, 1, 1])
        
        # Define output membership functions
        emotion_names = self.label_encoder.classes_
        for i in range(num_classes):
            # Create triangular membership functions centered around each class index
            lower = max(0, i-0.5)
            upper = min(num_classes-1, i+0.5)
            emotion_class[str(i)] = fuzz.trimf(emotion_class.universe, [lower, i, upper])
        
        # Create rules
        rules = []
        for i in range(num_classes):
            # When ML confidence is high, trust ML prediction
            rules.append(ctrl.Rule(ml_confidence['high'], emotion_class[str(i)]))
            
            # When term similarity is high but ML confidence is low/medium, trust term similarity
            rules.append(ctrl.Rule(
                ml_confidence['low'] & term_similarity['high'], 
                emotion_class[str(i)]
            ))
            rules.append(ctrl.Rule(
                ml_confidence['medium'] & term_similarity['high'], 
                emotion_class[str(i)]
            ))
        
        # Create control system
        ctrl_system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(ctrl_system)
    
    def fit(self, X, y):
        # Preprocess text
        X_processed = [preprocess_text(text) for text in X]
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # Create document-term matrix
        X_vectorized = self.vectorizer.fit_transform(X_processed)
        
        # Train ML model
        self.ml_model.fit(X_vectorized, y_encoded)
        
        # Calculate term frequencies for each class
        num_classes = len(self.label_encoder.classes_)
        for emotion in range(num_classes):
            emotion_docs = X_vectorized[y_encoded == emotion]
            if emotion_docs.shape[0] > 0:
                self.class_term_frequencies[emotion] = np.sum(emotion_docs, axis=0).A.flatten() / emotion_docs.shape[0]
            else:
                self.class_term_frequencies[emotion] = np.zeros(X_vectorized.shape[1])
        
        # Create fuzzy system
        self.fuzzy_system = self._create_fuzzy_system(num_classes)
        
        return self
    
    def predict(self, X):
        # Preprocess text
        X_processed = [preprocess_text(text) for text in X]
        
        # Create document-term matrix
        X_vectorized = self.vectorizer.transform(X_processed)
        
        # Get ML predictions and probabilities
        ml_predictions = self.ml_model.predict(X_vectorized)
        ml_probs = self.ml_model.predict_proba(X_vectorized)
        
        # Final predictions
        predictions = []
        
        for i in range(X_vectorized.shape[0]):
            # Get ML confidence (probability of predicted class)
            ml_pred = ml_predictions[i]
            ml_confidence = ml_probs[i][ml_pred]
            
            # Get term similarity for each emotion class
            sample = X_vectorized[i].toarray().flatten()
            similarity_scores = []
            
            for emotion in range(len(self.label_encoder.classes_)):
                emotion_freq = self.class_term_frequencies[emotion]
                dot_product = np.dot(sample, emotion_freq)
                magnitude1 = np.sqrt(np.sum(sample**2))
                magnitude2 = np.sqrt(np.sum(emotion_freq**2))
                
                if magnitude1 == 0 or magnitude2 == 0:
                    similarity = 0
                else:
                    similarity = dot_product / (magnitude1 * magnitude2)
                
                similarity_scores.append(similarity)
            
            max_similarity = max(similarity_scores)
            best_match_emotion = np.argmax(similarity_scores)
            
            # If ML confidence is high, use ML prediction
            if ml_confidence > 0.7:
                predictions.append(ml_pred)
            # If term similarity is high, use that
            elif max_similarity > 0.6:
                predictions.append(best_match_emotion)
            # Otherwise use fuzzy logic to decide
            else:
                try:
                    self.fuzzy_system.input['ml_confidence'] = ml_confidence
                    self.fuzzy_system.input['term_similarity'] = max_similarity
                    self.fuzzy_system.compute()
                    
                    # Get defuzzified output
                    emotion_idx = round(self.fuzzy_system.output['emotion_class'])
                    
                    # Ensure prediction is within valid range
                    emotion_idx = max(0, min(len(self.label_encoder.classes_)-1, emotion_idx))
                    predictions.append(emotion_idx)
                except:
                    # Fallback to ML if fuzzy system fails
                    predictions.append(ml_pred)
        
        return predictions

# Example usage
if __name__ == "__main__":
    print("Loading data...")
    train_data = pd.read_csv("data/train.csv")
    val_data = pd.read_csv("data/val.csv")
    test_data = pd.read_csv("data/test.csv")
    
    print("Training hybrid model...")
    model = HybridEmotionClassifier()
    model.fit(train_data["text"], train_data["label"])
    
    print("Evaluating on validation set...")
    val_pred = model.predict(val_data["text"])
    val_true = model.label_encoder.transform(val_data["label"])
    
    print("\nValidation Results:")
    val_accuracy = accuracy_score(val_true, val_pred)
    print(f"Accuracy: {val_accuracy:.4f}")
    print(classification_report(val_true, val_pred, target_names=model.label_encoder.classes_))
    
    print("\nEvaluating on test set...")
    test_pred = model.predict(test_data["text"])
    test_true = model.label_encoder.transform(test_data["label"])
    
    print("\nTest Results:")
    test_accuracy = accuracy_score(test_true, test_pred)
    print(f"Accuracy: {test_accuracy:.4f}")
    print(classification_report(test_true, test_pred, target_names=model.label_encoder.classes_))
    
    print("Done!")
