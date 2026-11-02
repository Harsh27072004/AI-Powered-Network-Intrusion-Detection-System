
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "random_forest_binary_v3.pkl"
INPUT_PATH = ROOT / "reports" / "live_capture_01_v3.csv"

print("Loading model and live traffic...")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(INPUT_PATH)

features = list(model.feature_names_in_)
X = df[features].apply(pd.to_numeric, errors="coerce")

# Validate feature values
if X.isna().any().any():
    raise ValueError("Missing or non-numeric feature values found.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Infinite feature values found.")

# Get attack probabilities
probabilities = model.predict_proba(X)

attack_class_index = list(model.classes_).index(1)
attack_probs = probabilities[:, attack_class_index]

print("\nTOTAL LIVE FLOWS:", len(df))
print("\nTHRESHOLD SENSITIVITY RESULTS")
print("-" * 55)

for threshold in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
    attacks = int((attack_probs >= threshold).sum())
    benign = len(df) - attacks

    print(
        f"Threshold: {threshold:.2f} | "
        f"Attack predictions: {attacks:3d} | "
        f"Benign predictions: {benign:3d}"
    )

print("\nAttack probability summary:")
print(pd.Series(attack_probs).describe().to_string())

print("\nNo model files or thresholds were changed.")
