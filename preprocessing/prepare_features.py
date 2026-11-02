import pandas as pd
import numpy as np
import os

# =========================================================
# SETTINGS
# =========================================================

INPUT_DIR = "dataset/processed/splits"
OUTPUT_DIR = "dataset/processed/prepared"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "X_train": "X_train.csv",
    "X_validation": "X_validation.csv",
    "X_test": "X_test.csv"
}

# Features that contain exactly the same information
# as another feature or contain no useful variation.
FEATURES_TO_REMOVE = [
    # Constant features
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",

    # Exact duplicate features
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Avg Fwd Segment Size",
    "Fwd Header Length.1",
    "ECE Flag Count"
]

# =========================================================
# PROCESS DATASETS
# =========================================================

print("=" * 70)
print("FINAL FEATURE PREPARATION")
print("=" * 70)

for dataset_name, filename in FILES.items():

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print("\n" + "-" * 70)
    print("Processing:", dataset_name)
    print("-" * 70)

    # Load dataset
    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    # -----------------------------------------------------
    # Replace infinity values
    # -----------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # Remove features
    # -----------------------------------------------------

    columns_to_remove = [
        column
        for column in FEATURES_TO_REMOVE
        if column in df.columns
    ]

    df = df.drop(
        columns=columns_to_remove
    )

    # -----------------------------------------------------
    # Handle remaining missing values
    # -----------------------------------------------------

    missing_before = df.isnull().sum().sum()

    if missing_before > 0:

        # Use median calculated separately for each dataset.
        # This is only a temporary safety step.
        df = df.fillna(df.median(numeric_only=True))

    missing_after = df.isnull().sum().sum()

    # -----------------------------------------------------
    # Check infinite values
    # -----------------------------------------------------

    infinite_count = np.isinf(
        df.select_dtypes(include=np.number)
    ).sum().sum()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    df.to_csv(
        output_path,
        index=False
    )

    print("Removed features:", len(columns_to_remove))
    print("Final shape:", df.shape)
    print("Missing values before:", missing_before)
    print("Missing values after:", missing_after)
    print("Infinite values:", infinite_count)

    print("Saved:", output_path)

print("\n" + "=" * 70)
print("FINAL FEATURE PREPARATION COMPLETE")
print("=" * 70)