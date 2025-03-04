import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import time
import os

# Import our custom modules
from fuzzy_emotion_detector import FuzzyEmotionDetector, optimize_fuzzy_system

def load_data():
    """Load and prepare the emotion dataset"""
    print("Loading emotion dataset...")
    from datasets import load_dataset
    
    # Load dataset
    ds = load_dataset("dair-ai/emotion", "split")
    
    # Define emotion labels
    emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    
    # Print dataset info
    print(f"Dataset loaded successfully!")
    print(f"Train set: {len(ds['train'])} samples")
    print(f"Validation set: {len(ds['validation'])} samples")
    print(f"Test set: {len(ds['test'])} samples")
    print(f"Emotion labels: {emotion_labels}\n")
    
    # Preprocess data
    def preprocess_text(text):
        return re.sub(r"[^a-zA-Z\s]", "", text.lower())
    
    print("Preprocessing dataset...")
    train_texts = [preprocess_text(text) for text in ds['train']['text']]
    train_labels = ds['train']['label']
    val_texts = [preprocess_text(text) for text in ds['validation']['text']]
    val_labels = ds['validation']['label']
    test_texts = [preprocess_text(text) for text in ds['test']['text']]
    test_labels = ds['test']['label']
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, emotion_labels

def train_model(train_texts, train_labels, val_texts, val_labels):
    """Train and optimize the fuzzy emotion detection model"""
    print("\n" + "="*50)
    print("Training Fuzzy Emotion Detection Model")
    print("="*50)
    
    # Train with default parameters first
    print("\nTraining initial model with default parameters...")
    start_time = time.time()
    detector = FuzzyEmotionDetector(n_features=150)
    detector.fit(train_texts, train_labels)
    training_time = time.time() - start_time
    print(f"Initial model trained in {training_time:.2f} seconds")
    
    # Display top features
    print("\nTop 10 features used by the model:")
    for i, feature in enumerate(detector.important_features[:10]):
        print(f"{i+1}. {feature}")
    
    # Evaluate initial model
    val_preds = detector.predict(val_texts[:100])  # Using subset for quick check
    initial_accuracy = accuracy_score(val_labels[:100], val_preds)
    print(f"\nInitial validation accuracy (on subset): {initial_accuracy:.4f}")
    
    # Optimize model
    print("\nOptimizing model by trying different feature counts...")
    print("This may take a few minutes...")
    
    # Limit to smaller subset for faster optimization during testing
    train_subset = min(5000, len(train_texts))
    val_subset = min(1000, len(val_texts))
    
    best_model, best_accuracy = optimize_fuzzy_system(
        train_texts[:train_subset],
        train_labels[:train_subset],
        val_texts[:val_subset],
        val_labels[:val_subset]
    )
    
    print(f"\nBest model achieved {best_accuracy:.4f} accuracy on validation subset")
    
    # Save model
    model_path = '/home/harish/code/Emotion/best_fuzzy_emotion_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {model_path}")
    
    return best_model

def evaluate_model(model, test_texts, test_labels, emotion_labels):
    """Evaluate model on test data and display results"""
    print("\n" + "="*50)
    print("Evaluating Model on Test Data")
    print("="*50)
    
    # Get predictions on test set (using subset for faster execution)
    test_subset = min(2000, len(test_texts))
    print(f"\nEvaluating model on {test_subset} test samples...")
    
    start_time = time.time()
    test_preds = model.predict(test_texts[:test_subset])
    inference_time = time.time() - start_time
    
    # Calculate accuracy
    test_accuracy = accuracy_score(test_labels[:test_subset], test_preds)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Average inference time per sample: {inference_time/test_subset*1000:.2f} ms")
    
    # Print classification report
    print("\nClassification Report:")
    emotion_names = [emotion_labels[i] for i in range(6)]
    report = classification_report(test_labels[:test_subset], test_preds, 
                                  target_names=emotion_names)
    print(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(test_labels[:test_subset], test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=emotion_names,
               yticklabels=emotion_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix plot
    plt.savefig('/home/harish/code/Emotion/confusion_matrix.png')
    print("Confusion matrix saved to '/home/harish/code/Emotion/confusion_matrix.png'")
    
    # If running in interactive environment, show the plot
    try:
        plt.show()
    except:
        pass

def demo_predictions(model, emotion_labels):
    """Demonstrate emotion predictions on sample texts"""
    print("\n" + "="*50)
    print("Emotion Prediction Demo")
    print("="*50)
    
    # Test with some examples
    test_examples = [
        "I am feeling so happy today!",
        "This makes me really angry!",
        "I'm terrified of what might happen next.",
        "I absolutely love this movie, it's amazing!",
        "What a surprise, I didn't expect that!",
        "I feel so sad about what happened.",
        "I'm so excited to see you tomorrow!",
        "I hate when people don't keep their promises.",
        "The news made me very anxious and worried."
    ]
    
    print("\nPredicting emotions for sample texts:")
    for example in test_examples:
        emotion_idx = model.predict_single(example)
        emotion = emotion_labels[emotion_idx]
        print(f"Text: '{example}'")
        print(f"Predicted emotion: {emotion}\n")
    
    # Interactive mode
    print("\nInteractive Mode:")
    print("Type a text to predict its emotion (or 'exit' to quit)")
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() == 'exit':
            break
        
        emotion_idx = model.predict_single(user_input)
        emotion = emotion_labels[emotion_idx]
        print(f"Predicted emotion: {emotion}")

def main():
    """Main function to run the complete emotion detection workflow"""
    print("="*50)
    print("Fuzzy Logic Emotion Detection System")
    print("="*50)
    
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, emotion_labels = load_data()
    
    # Check if model exists, load it or train a new one
    model_path = '/home/harish/code/Emotion/best_fuzzy_emotion_model.pkl'
    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    else:
        model = train_model(train_texts, train_labels, val_texts, val_labels)
    
    # Evaluate model
    evaluate_model(model, test_texts, test_labels, emotion_labels)
    
    # Demo predictions
    demo_predictions(model, emotion_labels)
    
    print("\nEmotion detection workflow completed!")

if __name__ == "__main__":
    main()
