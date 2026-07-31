"""
Data Validation Script
---------------------
This script performs the following steps:
1. Loads the source dataset from the raw data directory.
2. Validates the presence of expected columns in the dataset.
3. Performs basic data quality checks including missing values, duplicates, and data types.
4. Generates a validation summary with the status of each validation check.
5. Saves the validated dataset for downstream data preparation.
"""

import os
import sys
import pandas as pd

# Define source dataset location
source_file_path = "data/raw"
source_file_name = "tourism.csv"

# Define validated dataset location
validated_data_directory = "data/validated"

# Create validated data directory
os.makedirs(validated_data_directory, exist_ok=True)

validation_results = []

def log_validation_result(validation_name, status, details):
    validation_results.append({
        "Validation": validation_name,
        "Status": status,
        "Details": details
    })

def print_validation_summary():
    print("\nValidation Summary:")
    print("-" * 60)

    for result in validation_results:
        print(
            f"{result['Validation']:<25}: "
            f"{result['Status']:<5} - "
            f"{result['Details']}"
        )

try:
    file_path   = os.path.join(source_file_path,source_file_name)
    tourism_df  = pd.read_csv(file_path)

    log_validation_result(
        "Source file check",
        "PASS",
        f"Source file '{source_file_name}' loaded successfully from '{source_file_path}'."
    )

    expected_source_columns = [
        'Unnamed: 0',
        'CustomerID',
        'ProdTaken',
        'Age',
        'TypeofContact',
        'CityTier',
        'DurationOfPitch',
        'Occupation',
        'Gender',
        'NumberOfPersonVisiting',
        'NumberOfFollowups',
        'ProductPitched',
        'PreferredPropertyStar',
        'MaritalStatus',
        'NumberOfTrips',
        'Passport',
        'PitchSatisfactionScore',
        'OwnCar',
        'NumberOfChildrenVisiting',
        'Designation',
        'MonthlyIncome'
    ]

    actual_source_columns = tourism_df.columns.tolist()

    missing_columns = list(
        set(expected_source_columns) -
        set(actual_source_columns)
    )

    extra_columns = list(
        set(actual_source_columns) -
        set(expected_source_columns)
    )

    if missing_columns:
        log_validation_result(
            "Missing columns check",
            "FAIL",
            f"Missing required columns: {missing_columns}."
        )

        print_validation_summary()
        sys.exit("Validation failed: Required columns are missing.")

    else:
        log_validation_result(
            "Missing columns check",
            "PASS",
            "All expected columns are available in the dataset."
        )

    if extra_columns:
        log_validation_result(
            "Extra columns check",
            "FAIL",
            f"Unexpected columns found: {extra_columns}."
        )
        
        print_validation_summary()
        
        sys.exit("Validation failed: Unexpected columns are present.")

    else:
        log_validation_result(
            "Extra columns check",
            "PASS",
            "No unexpected columns were found in the dataset."
        )

    if tourism_df.empty:
        log_validation_result(
            "Dataset size check",
            "FAIL",
            "Dataset does not contain any records."
        )

        print_validation_summary()
        sys.exit("Validation failed: Dataset is empty.")

    else:
        log_validation_result(
            "Dataset size check",
            "PASS",
            f"Dataset contains {tourism_df.shape[0]:,} records and {tourism_df.shape[1]} columns."
        )

    dtype_summary = []

    for dtype, cols in tourism_df.columns.to_series().groupby(tourism_df.dtypes):
        dtype_summary.append(
            f"{dtype}: {list(cols)}"
        )

    log_validation_result(
        "Data types check",
        "PASS",
        "Data types validated successfully. " + " | ".join(dtype_summary)
    )

    null_records_count = tourism_df.isnull().sum().sum()

    if null_records_count > 0:
        log_validation_result(
            "Null data check",
            "FAIL",
            f"{null_records_count:,} missing values detected in the dataset."
        )

    else:
        log_validation_result(
            "Null data check",
            "PASS",
            "No missing values were detected in the dataset."
        )

    duplicate_data_count = tourism_df.duplicated().sum()

    if duplicate_data_count > 0:
        tourism_df = tourism_df.drop_duplicates()

        log_validation_result(
            "Duplicate data check",
            "PASS",
            f"{duplicate_data_count:,} duplicate records detected and removed."
        )

    else:
        log_validation_result(
            "Duplicate data check",
            "PASS",
            "No duplicate records were detected in the dataset."
        )

    validated_file_name = "validated_tourism.csv"

    validated_dataset_path = os.path.join(
        validated_data_directory,
        validated_file_name
    )

    tourism_df.to_csv(
        validated_dataset_path,
        index=False
    )

    print(
        f"\nValidated dataset '{validated_file_name}' "
        f"saved successfully at '{validated_data_directory}'."
    )

except FileNotFoundError:

    log_validation_result(
        "Source file check",
        "FAIL",
        f"Source file '{source_file_name}' not found at '{source_file_path}'."
    )

    print_validation_summary()

    print("\nValidation failed: Source dataset is missing.")


except Exception as e:
    log_validation_result(
        "Unexpected error",
        "FAIL",
        str(e)
    )

    print_validation_summary()
    print("\nValidation failed due to an unexpected error.")
    sys.exit(1)


else:
    print_validation_summary()
