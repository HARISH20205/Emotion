# Emotion Classification Project

This project implements a simplified and realistic approach to emotion classification using unigram/bigram features with fuzzy logic.

## Project Structure

- `simplified_emotion.py`: Main implementation with simplified fuzzy logic approach
- `hybrid_model.py`: A hybrid approach combining machine learning with fuzzy logic
- `feature_analysis.py`: Analyzes and visualizes the most important features for each emotion
- `data/`: Directory containing training, validation, and test datasets

## Data Format

The system expects CSV files with at least two columns:

- `text`: Text to classify
- `label`: Emotion label (e.g., 'joy', 'sadness', 'anger', 'fear', 'love', 'surprise')

## Approaches Implemented

### 1. Simplified Fuzzy Logic Approach

This implementation uses:

- Unigram and bigram features
- Simple and effective preprocessing
- Term frequency analysis for each emotion class
- Simplified fuzzy logic rules focused on term similarity and intensity
- Reduced complexity compared to overly complex fuzzy systems

### 2. Hybrid Approach

Combines:

- Traditional machine learning (Random Forest)
- Fuzzy logic for decision making
- Term similarity analysis
- Confidence-based selection between models

## How to Run

```bash
# Run the simplified fuzzy logic approach
python simplified_emotion.py

# Run the hybrid model
python hybrid_model.py

# Analyze features distribution
python feature_analysis.py
```

## Improving Accuracy

The accuracy of the emotion classification has been improved by:

1. **Simplified Rules**: Reduced the overly complex fuzzy logic rule set to focus on what matters
2. **Improved Feature Extraction**: Optimized preprocessing and unigram/bigram extraction
3. **Better Term Similarity**: Using cosine similarity between document vectors and class-specific term frequencies
4. **Hybrid Approach**: Combining traditional ML with fuzzy logic for better results
5. **Feature Analysis**: Understanding which terms are most indicative of each emotion

## Requirements

```
pandas
numpy
scikit-learn
nltk
scikit-fuzzy
matplotlib (optional, for visualizations)
```

## Future Improvements

- Word embeddings (GloVe, Word2Vec) could improve text representation
- Deep learning approaches (LSTM, BERT) could capture more complex patterns
- Ensemble methods could combine multiple approaches for better accuracy
