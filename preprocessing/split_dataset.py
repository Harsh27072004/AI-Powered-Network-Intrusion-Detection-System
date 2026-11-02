import pandas as pd
from sklearn.model_selection import train_test_split
import os

# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "dataset/processed/binary_nids_dataset.csv"
OUTPUT_DIR = "dataset/processed/splits"

RANDOM_STATE = 42

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("LOADING BINARY NIDS DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("Total records:", len(df))
print("Total columns:", len(df.columns))

# =========================================================
# SEPARATE FEATURES AND TARGET
# =========================================================

# Label is text describing the original attack type.
# We don't want the model to use it as an input feature.

X = df.drop(columns=["Label", "Binary_Label"])

# Binary_Label is our target:
# 0 = BENIGN
# 1 = ATTACK

y = df["Binary_Label"]

print("\nFeatures:", X.shape[1])
print("Target records:", len(y))

# =========================================================
# FIRST SPLIT
# 70% TRAINING
# 30% TEMPORARY
# =========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================================================
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# =========================================================

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

# =========================================================
# DISPLAY SIZES
# =========================================================

print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print("\nTraining:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_validation:", X_validation.shape)
print("y_validation:", y_validation.shape)

print("\nTesting:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining:")
print(y_train.value_counts())

print("\nValidation:")
print(y_validation.value_counts())

print("\nTesting:")
print(y_test.value_counts())

# =========================================================
# SAVE DATASETS
# =========================================================

print("\n" + "=" * 70)
print("SAVING SPLITS")
print("=" * 70)

X_train.to_csv(
    f"{OUTPUT_DIR}/X_train.csv",
    index=False
)

X_validation.to_csv(
    f"{OUTPUT_DIR}/X_validation.csv",
    index=False
)

X_test.to_csv(
    f"{OUTPUT_DIR}/X_test.csv",
    index=False
)

y_train.to_csv(
    f"{OUTPUT_DIR}/y_train.csv",
    index=False
)

y_validation.to_csv(
    f"{OUTPUT_DIR}/y_validation.csv",
    index=False
)

y_test.to_csv(
    f"{OUTPUT_DIR}/y_test.csv",
    index=False
)

print("\nFiles saved successfully.")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("DATASET SPLITTING COMPLETE")
print("=" * 70)