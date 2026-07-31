"""
Model Training Script
---------------------
This script performs the following steps:
1. Loads the processed training and testing datasets.
2. Creates preprocessing pipelines for numerical and categorical features.
3. Performs hyperparameter tuning using multiple model configurations.
4. Tracks model experiments using MLflow.
5. Saves the best-performing model and its hyperparameters.
"""

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import xgboost as xgb

from itertools import product

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


# Define paths
processed_data_directory = "data/processed"
trained_model_directory = "trained_models"
model_file_name = "wellness_model.joblib"

os.makedirs(trained_model_directory, exist_ok=True)

print(f"Trained models directory '{trained_model_directory}' created successfully!!")


# Step 1: Load datasets

X_train = pd.read_csv(
    os.path.join(processed_data_directory, "X_train.csv")
)

X_test = pd.read_csv(
    os.path.join(processed_data_directory, "X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(processed_data_directory, "y_train.csv")
).squeeze()

y_test = pd.read_csv(
    os.path.join(processed_data_directory, "y_test.csv")
).squeeze()

print("Training and testing datasets loaded successfully!!")
print(f"Training dataset shape: {X_train.shape}")
print(f"Testing dataset shape: {X_test.shape}")


# Step 2: Define features

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

numerical_features = [
    "Age",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "NumberOfTrips",
    "MonthlyIncome"
]

ordinal_features = [
    "CityTier",
    "PreferredPropertyStar",
    "PitchSatisfactionScore",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting"
]

print("Feature groups identified successfully!!")


# Step 3: Handle class imbalance

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

scale_pos_weight = class_weights[1] / class_weights[0]

print(f"Scale positive weight: {scale_pos_weight:.2f}")


# Step 4: Create preprocessing pipeline

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessing_pipeline = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("ordinal", ordinal_pipeline, ordinal_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("Preprocessing pipeline created successfully!!")


# Step 5: Define hyperparameter grid
hyperparameter_grid = {
    "n_estimators": [100, 150, 200, 250],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "subsample": [0.7, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0]
}

total_experiments = np.prod(
    [len(values) for values in hyperparameter_grid.values()]
)

print(f"Total experiment runs: {total_experiments}")


# Step 6: Configure MLflow

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment(
    "visit_with_us_customer_purchase_prediction"
)


# Step 7: Train models

best_f1_score = 0
best_model = None
best_hyperparameters = None


for experiment_number, parameter_values in enumerate(
    product(*hyperparameter_grid.values()),
    start=1
):

    current_hyperparameters = dict(
        zip(
            hyperparameter_grid.keys(),
            parameter_values
        )
    )

    print(
        f"Experiment {experiment_number}/{total_experiments}: "
        f"{current_hyperparameters}"
    )

    with mlflow.start_run():

        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            **current_hyperparameters
        )

        model_pipeline = make_pipeline(
            preprocessing_pipeline,
            model
        )

        model_pipeline.fit(
            X_train,
            y_train
        )

        train_predictions = model_pipeline.predict(X_train)
        test_predictions = model_pipeline.predict(X_test)

        train_report = classification_report(
            y_train,
            train_predictions,
            output_dict=True
        )

        test_report = classification_report(
            y_test,
            test_predictions,
            output_dict=True
        )

        mlflow.log_params(
            current_hyperparameters
        )

        mlflow.log_metrics({
            "train_accuracy": train_report["accuracy"],
            "train_f1_score": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1_score": test_report["1"]["f1-score"]
        })

        if test_report["1"]["f1-score"] > best_f1_score:

            best_f1_score = test_report["1"]["f1-score"]
            best_model = model_pipeline
            best_hyperparameters = current_hyperparameters


# Step 8: Save best model

if best_model is not None:

    model_path = os.path.join(
        trained_model_directory,
        model_file_name
    )

    joblib.dump(
        best_model,
        model_path
    )

    parameter_path = os.path.join(
        trained_model_directory,
        "best_model_parameters.json"
    )

    with open(parameter_path, "w") as file:
        json.dump(
            best_hyperparameters,
            file,
            indent=4
        )

    print(f"Best model saved: {model_path}")
    print(f"Best parameters: {best_hyperparameters}")
    print(f"Best F1 Score: {best_f1_score:.4f}")


    with mlflow.start_run(
        run_name="Best_Model"
    ):

        mlflow.log_metric(
            "best_validation_f1_score",
            best_f1_score
        )

        mlflow.sklearn.log_model(
            best_model,
            "best_model"
        )

    print("Best model logged to MLflow successfully!!")
