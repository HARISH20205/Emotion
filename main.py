import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import load_and_preprocess_data
from emotion_model import EmotionClassifier

# Ensure data directory exists
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")
    print("Please place your train.csv, val.csv, and test.csv files in the data directory.")
    exit()

# Paths to data files
train_path = os.path.join(data_dir, "train.csv")
val_path = os.path.join(data_dir, "val.csv")
test_path = os.path.join(data_dir, "test.csv")

# Load and preprocess data
train_data, val_data, test_data, train_raw, val_raw, test_raw = load_and_preprocess_data(
    train_path, val_path, test_path
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Initialize and train the model
print("\nTraining SVM model...")
svm_model = EmotionClassifier(model_type='svm')
svm_model.train(train_data["text"], train_data["label"])

# Evaluate on validation set
val_accuracy, val_report = svm_model.evaluate(val_data["text"], val_data["label"])
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation Report:\n{val_report}")

# Evaluate on test set
test_accuracy, test_report = svm_model.evaluate(test_data["text"], test_data["label"])
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Report:\n{test_report}")

# Train Random Forest model for comparison
print("\nTraining Random Forest model for comparison...")
rf_model = EmotionClassifier(model_type='rf')
rf_model.train(train_data["text"], train_data["label"])

# Evaluate RF model
rf_val_accuracy, _ = rf_model.evaluate(val_data["text"], val_data["label"])
rf_test_accuracy, _ = rf_model.evaluate(test_data["text"], test_data["label"])

print(f"\nRandom Forest Validation Accuracy: {rf_val_accuracy:.4f}")
print(f"Random Forest Test Accuracy: {rf_test_accuracy:.4f}")

# Hyperparameter tuning (uncomment to run - can take time)
print("\nPerforming hyperparameter tuning on SVM model...")
svm_model.tune_hyperparameters(train_data["text"], train_data["label"])

# Final evaluation with tuned model
print("\nEvaluating tuned model...")
final_val_accuracy, _ = svm_model.evaluate(val_data["text"], val_data["label"])
final_test_accuracy, final_test_report = svm_model.evaluate(test_data["text"], test_data["label"])

print(f"\nFinal Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
print(f"Final Test Report:\n{final_test_report}")

# Visualize confusion matrix
y_true = svm_model.label_encoder.transform(test_data["label"])
y_pred = svm_model.predict(test_data["text"])

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=svm_model.label_encoder.classes_
)

plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title('Emotion Classification Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print("\nConfusion matrix saved as 'confusion_matrix.png'")
print("\nAnalysis complete!")
