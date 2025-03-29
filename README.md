# Emotion Detection Model

This project implements a machine learning pipeline for classifying text into emotion categories.

## Project Structure

- `main.py`: Main script to run the complete pipeline
- `emotion_model.py`: Contains the EmotionClassifier class implementation
- `utils.py`: Contains preprocessing and utility functions
- `data/`: Directory for storing datasets

## Setup and Requirements

1. Install required dependencies:

   ```
   pip install pandas numpy scikit-learn nltk matplotlib
   ```

2. Place your datasets in the `data/` directory:

   - `train.csv`: Training data
   - `val.csv`: Validation data
   - `test.csv`: Test data

   Each CSV file should contain at least two columns:

   - `text`: The text to classify
   - `label`: The emotion label

## Running the Code

```bash
python main.py
```

This will:

1. Load and preprocess the data
2. Train an SVM model and a Random Forest model
3. Evaluate both models on validation and test sets
4. Perform hyperparameter tuning
5. Generate a confusion matrix visualization

## Implementation Notes

The implementation uses:

- TF-IDF vectorization for feature extraction
- Support Vector Machine (LinearSVC) as the primary classifier
- Random Forest as a comparison model
- GridSearchCV for hyperparameter tuning

## Extending the Model

To extend the model:

1. Add new preprocessing techniques in `utils.py`
2. Implement additional classifiers in `emotion_model.py`
3. Modify `main.py` to use the new implementations
