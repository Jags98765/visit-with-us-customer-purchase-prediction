
# Visit With Us - Wellness Tourism Package Prediction

## Overview

An end-to-end **MLOps project** to predict whether a customer will purchase the **Wellness Tourism Package** using customer demographics and interaction data.

The project implements a complete machine learning workflow including:

- Data validation and preprocessing
- XGBoost model training and tuning
- MLflow experiment tracking
- Model evaluation
- Streamlit deployment
- GitHub Actions CI/CD automation

---

## Business Problem

Visit With Us aims to identify customers who are more likely to purchase the Wellness Tourism Package before marketing outreach.

The machine learning model helps improve customer targeting, optimize marketing campaigns, and support data-driven business decisions.

**Target Variable:** `ProdTaken`

- `0` → Customer will not purchase the package
- `1` → Customer will purchase the package

---

## Project Structure

```text
visit-with-us-customer-purchase-prediction
│
├── .github/
│   └── workflows/
│       └── pipeline.yml
│
├── data/
│   ├── raw/
│   ├── validated/
│   └── processed/
│
├── pipeline_components/
│   ├── data_validation/
│   ├── data_preparation/
│   ├── model_training/
│   └── model_evaluation/
│
├── trained_models/
│   └── wellness_model.joblib
│
├── model_deployment/
│   └── streamlit_app.py
│
├── requirements.txt
├── requirements_pipeline.txt
└── README.md


---

### Part 4 — ML Workflow

```markdown
---

## ML Workflow

### 1. Data Validation

- Validates dataset structure and required columns.
- Checks missing values and duplicate records.
- Stores validated data for further processing.

### 2. Data Preparation

- Removes unnecessary identifier columns.
- Cleans and transforms customer data.
- Creates training and testing datasets.

### 3. Model Training

- Uses XGBoost Classification algorithm.
- Performs hyperparameter tuning.
- Tracks experiments using MLflow.
- Saves the best-performing model.

### 4. Model Evaluation

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

### 5. Deployment

A Streamlit application provides an interactive interface for customer purchase prediction.

### 6. CI/CD Pipeline

GitHub Actions automates:

- Data validation
- Data preparation
- Model training
- Model evaluation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- MLflow
- Streamlit
- GitHub Actions

---

## Run Project

Install dependencies:

```bash
pip install -r requirements_pipeline.txt

python pipeline_components/data_validation/validate_source_data.py

python pipeline_components/data_preparation/prepare_data.py

python pipeline_components/model_training/train_model.py

python pipeline_components/model_evaluation/evaluate_model.py

streamlit run model_deployment/streamlit_app.py


---

### Part 6 — Deployment Links and Conclusion

```markdown
---

## Deployment Links

GitHub Repository:

https://github.com/Jags98765/visit-with-us-customer-purchase-prediction/

Streamlit Application:

https://visit-with-us-customer-purchase-prediction.streamlit.app/

---

## Conclusion

This MLOps solution automates customer purchase prediction and enables Visit With Us to improve customer targeting through machine learning and data-driven marketing strategies.
