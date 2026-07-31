
"""
Model Evaluation Script
-----------------------
This script performs the following steps:
1. Loads the processed testing dataset.
2. Loads the trained machine learning model.
3. Generates predictions using the trained model.
4. Evaluates model performance using classification metrics.
5. Displays the final evaluation results.
"""

# Import required libraries
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Step 1: Load testing datasets

processed_data_directory = "data/processed"
trained_model_directory = "trained_models"

model_file_name = "wellness_model.joblib"

X_test = pd.read_csv(
    os.path.join(processed_data_directory, "X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(processed_data_directory, "y_test.csv")
).squeeze()

print("Testing dataset loaded successfully!!")
print(f"Testing dataset shape: {X_test.shape}")


# Step 2: Load the Trained Model
# Define trained model path
model_path = os.path.join(
    trained_model_directory,
    model_file_name
)

# Load the trained model
trained_model = joblib.load(model_path)

print(f"Trained model loaded successfully from '{model_path}'")

# Step 3: Generate Predictions and Evaluate Model Performance
# Generate predictions using the trained model
y_predictions = trained_model.predict(X_test)

print("Predictions generated successfully!!")


# Calculate evaluation metrics
accuracy = accuracy_score(
    y_test,
    y_predictions
)

precision = precision_score(
    y_test,
    y_predictions
)

recall = recall_score(
    y_test,
    y_predictions
)

f1 = f1_score(
    y_test,
    y_predictions
)


# Display model performance
print("\nModel Performance Summary")
print("-" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predictions))


# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predictions))
