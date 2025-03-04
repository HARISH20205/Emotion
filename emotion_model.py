import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from utils import preprocess_text

class EmotionClassifier:
    def __init__(self, model_type='svm'):
        self.model_type = model_type
        self.label_encoder = LabelEncoder()
        
        # Choose model based on type
        if model_type == 'svm':
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)),
                ('classifier', LinearSVC(random_state=42))
            ])
        else:  # Default to Random Forest
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
    
    def train(self, X_train, y_train):
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Train the model
        self.pipeline.fit(X_train, y_encoded)
        return self
    
    def predict(self, X_test):
        # Predict and return encoded predictions
        return self.pipeline.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        # Encode test labels
        y_encoded = self.label_encoder.transform(y_test)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        report = classification_report(
            y_encoded, 
            y_pred, 
            target_names=[str(label) for label in self.label_encoder.classes_]
        )
        
        return accuracy, report

    def tune_hyperparameters(self, X_train, y_train):
        """Tune hyperparameters using grid search"""
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        if self.model_type == 'svm':
            param_grid = {
                'tfidf__max_features': [5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__C': [0.1, 1.0, 10.0]
            }
        else:
            param_grid = {
                'tfidf__max_features': [5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20]
            }
            
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_encoded)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Update pipeline with best parameters
        self.pipeline = grid_search.best_estimator_
        return self
