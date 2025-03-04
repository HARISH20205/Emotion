import pickle
import sys

def load_model(model_path='/home/harish/code/Emotion/best_fuzzy_emotion_model.pkl'):
    """Load the trained fuzzy emotion detection model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_emotion(text, model=None):
    """Predict the emotion of the given text"""
    if model is None:
        model = load_model()
    
    emotion_idx = model.predict_single(text)
    emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    return emotion_labels[emotion_idx]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Get text from command line
        text = ' '.join(sys.argv[1:])
        model = load_model()
        emotion = predict_emotion(text, model)
        print(f"Text: '{text}'")
        print(f"Predicted emotion: {emotion}")
    else:
        print("Please provide text for emotion prediction.")
        print("Usage: python emotion_prediction.py \"Your text here\"")
