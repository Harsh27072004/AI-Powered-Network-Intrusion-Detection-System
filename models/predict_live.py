import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# AI-NIDS LIVE PREDICTION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_binary.pkl"
)

LIVE_DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "live_flows.csv"
)

TRAIN_DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "prepared",
    "X_train.csv"
)


print("=" * 70)
print("AI-NIDS LIVE PCAP DETECTION")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD LIVE FEATURES
# ============================================================

print("\nLoading extracted network flows...")

live_df = pd.read_csv(LIVE_DATA_PATH)

print(f"Flows loaded: {len(live_df)}")
print(f"Features loaded: {len(live_df.columns)}")


# ============================================================
# GET EXACT TRAINING FEATURE ORDER
# ============================================================

print("\nChecking feature compatibility...")

train_sample = pd.read_csv(
    TRAIN_DATA_PATH,
    nrows=1
)

feature_columns = list(train_sample.columns)


# Check missing features
missing_features = [
    col for col in feature_columns
    if col not in live_df.columns
]

# Check extra features
extra_features = [
    col for col in live_df.columns
    if col not in feature_columns
]


if missing_features:
    print("\nERROR: Missing features:")
    for feature in missing_features:
        print(" -", feature)
    raise SystemExit(1)


if extra_features:
    print("\nERROR: Extra features:")
    for feature in extra_features:
        print(" -", feature)
    raise SystemExit(1)


# Reorder exactly like training data
X_live = live_df[feature_columns].copy()


# ============================================================
# DATA CLEANING
# ============================================================

print("\nChecking live feature values...")

X_live = X_live.replace(
    [np.inf, -np.inf],
    np.nan
)

X_live = X_live.fillna(0)

print("NaN values after cleaning:",
      X_live.isna().sum().sum())

print("Infinite values after cleaning:",
      np.isinf(X_live.select_dtypes(include=np.number)).sum().sum())


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_live)

probabilities = model.predict_proba(X_live)


# Find probability column corresponding to ATTACK = 1
attack_index = list(model.classes_).index(1)

attack_probability = probabilities[:, attack_index]
benign_probability = 1 - attack_probability


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LIVE DETECTION RESULTS")
print("=" * 70)

for i in range(len(X_live)):

    prediction = predictions[i]

    if prediction == 1:
        label = "ATTACK"
        confidence = attack_probability[i]
    else:
        label = "BENIGN"
        confidence = benign_probability[i]

    print(
        f"Flow {i + 1:3d}  ->  "
        f"{label:6s}  "
        f"Confidence: {confidence * 100:6.2f}%  "
        f"(BENIGN: {benign_probability[i] * 100:6.2f}% | "
        f"ATTACK: {attack_probability[i] * 100:6.2f}%)"
    )


# ============================================================
# SUMMARY
# ============================================================

benign_count = int(np.sum(predictions == 0))
attack_count = int(np.sum(predictions == 1))

print("\n" + "=" * 70)
print("DETECTION SUMMARY")
print("=" * 70)

print(f"Total flows : {len(predictions)}")
print(f"BENIGN      : {benign_count}")
print(f"ATTACK      : {attack_count}")

print("=" * 70)
print("LIVE DETECTION COMPLETE")
print("=" * 70)