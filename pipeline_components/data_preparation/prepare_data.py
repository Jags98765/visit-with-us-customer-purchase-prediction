"""
Data Preparation Script
-----------------------
This script performs the following steps:
1. Loads the validated dataset from the validated data directory.
2. Removes unnecessary columns from the dataset.
3. Splits the dataset into training and testing sets.
4. Saves the processed datasets to the processed data directory.
"""

import  os
import  pandas as pd
from    sklearn.model_selection import train_test_split


# Define dataset location
data_directory = "data/validated"
validated_file_name = "validated_tourism.csv"
validated_file_path = os.path.join(data_directory, validated_file_name)

# Load the validated dataset
dataset = pd.read_csv(validated_file_path)

print(f"Validated dataset '{validated_file_name}' loaded successfully from '{data_directory}'")
print(f"Dataset shape: {dataset.shape}")

# ------------------------------------------------------------------------------
# `CustomerID` and `Unnamed: 0` are identifier columns used only for record
# identification and are removed during data preparation as they do not provide
# predictive value for the machine learning model.
# ------------------------------------------------------------------------------
columns_to_remove = ["Unnamed: 0", "CustomerID"]

dataset = dataset.drop(columns=columns_to_remove)
print(f"Removed unnecessary columns: {columns_to_remove}")

# Fix inconsistent gender labels by updating 'Fe Male' to 'Female'
dataset['Gender'] = dataset['Gender'].replace('Fe Male', 'Female')

# Split the dataset into training and testing sets
target_column = "ProdTaken"

features = dataset.drop(columns=[target_column])
target = dataset[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42,
    stratify=target
)

print("Training and testing datasets created successfully!!")
print(f"Training dataset shape: {X_train.shape}")
print(f"Testing dataset shape: {X_test.shape}")

print("\nTarget distribution (Training Dataset):")
print(y_train.value_counts(normalize=True))

print("\nTarget distribution (Testing Dataset):")
print(y_test.value_counts(normalize=True))


# Create processed data directory
processed_data_directory = "data/processed"

os.makedirs(processed_data_directory, exist_ok=True)
print(f"Processed Data directory '{processed_data_directory}' created successfully!!")


# Save processed datasets
X_train.to_csv(os.path.join(processed_data_directory, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_data_directory, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_data_directory, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_data_directory, "y_test.csv"), index=False)

print(f"\nProcessed datasets 'X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv' saved successfully to '{processed_data_directory}'")
