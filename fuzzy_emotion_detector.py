import re
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import nltk
from collections import Counter
import random

# Download NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

class FuzzyEmotionDetector:
    def __init__(self, n_features=100):
        self.n_features = n_features
        self.stopwords = set(stopwords.words('english'))
        # Include stopwords removal in the vectorizer
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=self.stopwords
        )
        self.feature_selector = SelectKBest(chi2, k=n_features)
        self.scaler = MinMaxScaler()
        self.emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        self.important_features = []
        self.control_system = None
        # Add emotion keywords for better classification
        self.emotion_keywords = {
            0: ['sad', 'unhappy', 'depressed', 'miserable', 'grief'],  # sadness
            1: ['happy', 'joy', 'excited', 'delighted', 'glad'],       # joy
            2: ['love', 'adore', 'cherish', 'affection', 'passion'],   # love
            3: ['angry', 'mad', 'furious', 'annoyed', 'irritated'],    # anger
            4: ['afraid', 'fear', 'scared', 'terrified', 'anxious'],   # fear
            5: ['surprised', 'shocked', 'amazed', 'astonished', 'wow'] # surprise
        }

    def preprocess_text(self, text):
        """Clean and normalize text with stopword removal"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)
    
    def extract_features(self, texts):
        """Extract unigram and bigram features"""
        return self.vectorizer.transform(texts)
    
    def fit(self, train_texts, train_labels):
        """Fit the fuzzy emotion detector"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in train_texts]
        
        # Extract features
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Select top features
        X_selected = self.feature_selector.fit_transform(X, train_labels)
        
        # Get important feature names
        feature_indices = self.feature_selector.get_support(indices=True)
        feature_names = np.array(self.vectorizer.get_feature_names_out())[feature_indices]
        self.important_features = feature_names
        
        # Scale features to [0, 1]
        X_scaled = self.scaler.fit_transform(X_selected.toarray())
        
        # Build the fuzzy control system
        self._build_fuzzy_system()
        
        return self
    
    def _build_fuzzy_system(self):
        """Build the fuzzy inference system"""
        # Create antecedents (features)
        feature_vars = {}
        for i in range(min(5, len(self.important_features))):  # Use top 5 features max for simplicity
            name = f"feature_{i}"
            feature_vars[name] = ctrl.Antecedent(np.linspace(0, 1, 100), name)
            feature_vars[name]['low'] = fuzz.trapmf(feature_vars[name].universe, [0, 0, 0.3, 0.5])
            feature_vars[name]['medium'] = fuzz.trimf(feature_vars[name].universe, [0.3, 0.7, 0.9])
            feature_vars[name]['high'] = fuzz.trapmf(feature_vars[name].universe, [0.7, 0.9, 1, 1])
        
        # Create consequent (emotion)
        emotion = ctrl.Consequent(np.linspace(0, 5, 60), "emotion")
        
        # Define membership functions for each emotion
        emotion['sadness'] = fuzz.trimf(emotion.universe, [-0.5, 0, 0.5])
        emotion['joy'] = fuzz.trimf(emotion.universe, [0.5, 1, 1.5])
        emotion['love'] = fuzz.trimf(emotion.universe, [1.5, 2, 2.5])
        emotion['anger'] = fuzz.trimf(emotion.universe, [2.5, 3, 3.5])
        emotion['fear'] = fuzz.trimf(emotion.universe, [3.5, 4, 4.5])
        emotion['surprise'] = fuzz.trimf(emotion.universe, [4.5, 5, 5.5])
        
        # Define rules - simplified rules for demonstration
        rules = []
        
        # Sample rules - in a real system, these would be more sophisticated and based on data analysis
        if len(feature_vars) >= 1:
            rules.append(ctrl.Rule(feature_vars['feature_0']['high'], emotion['joy']))
            rules.append(ctrl.Rule(feature_vars['feature_0']['low'], emotion['sadness']))
        
        if len(feature_vars) >= 2:
            rules.append(ctrl.Rule(feature_vars['feature_1']['high'], emotion['love']))
            rules.append(ctrl.Rule(feature_vars['feature_1']['low'], emotion['anger']))
        
        if len(feature_vars) >= 3:
            rules.append(ctrl.Rule(feature_vars['feature_2']['high'], emotion['surprise']))
            rules.append(ctrl.Rule(feature_vars['feature_2']['low'], emotion['fear']))
        
        if len(feature_vars) >= 4:
            rules.append(ctrl.Rule(feature_vars['feature_3']['high'] & feature_vars['feature_0']['high'], emotion['joy']))
            rules.append(ctrl.Rule(feature_vars['feature_3']['low'] & feature_vars['feature_1']['low'], emotion['anger']))
        
        if len(feature_vars) >= 5:
            rules.append(ctrl.Rule(feature_vars['feature_4']['high'] & feature_vars['feature_2']['medium'], emotion['surprise']))
            rules.append(ctrl.Rule(feature_vars['feature_4']['low'] & feature_vars['feature_0']['medium'], emotion['love']))
        
        # Create control system
        self.control_system = ctrl.ControlSystem(rules)
        self.feature_vars = feature_vars
        
    def predict_single(self, text):
        """Predict emotion for a single text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract features
        X = self.vectorizer.transform([processed_text])
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected.toarray()).flatten()
        
        # Create a simulation
        sim = ctrl.ControlSystemSimulation(self.control_system)
        
        # Set input values
        for i, name in enumerate(self.feature_vars):
            if i < len(X_scaled):
                sim.input[name] = X_scaled[i]
        
        # Compute
        try:
            sim.compute()
            emotion_value = sim.output['emotion']
            # Round to nearest integer and ensure it's within bounds
            emotion_index = max(0, min(5, round(emotion_value)))
            return emotion_index
        except:
            # Default to most common emotion if computation fails
            return 1  # Joy as default
    
    def predict(self, texts):
        """Predict emotions for multiple texts"""
        return [self.predict_single(text) for text in texts]

def optimize_fuzzy_system(train_texts, train_labels, val_texts, val_labels):
    """
    Function to optimize the fuzzy system by trying different feature counts
    and returning the best performing model
    """
    from sklearn.metrics import accuracy_score
    
    best_accuracy = 0
    best_model = None
    
    for n_features in [50, 100, 200, 300]:
        detector = FuzzyEmotionDetector(n_features=n_features)
        detector.fit(train_texts, train_labels)
        
        # Predict on validation set
        val_preds = detector.predict(val_texts)
        accuracy = accuracy_score(val_labels, val_preds)
        
        print(f"Model with {n_features} features achieved {accuracy:.4f} accuracy")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = detector
    
    return best_model, best_accuracy
