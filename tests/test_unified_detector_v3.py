
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.unified_detector import UnifiedNIDSDetector

DATA = ROOT / "dataset" / "processed" / "prepared_v3"

print("Loading 100 real V3 test records...")
X = pd.read_csv(DATA / "X_test.csv", nrows=100)

print("Initializing detector...")
detector = UnifiedNIDSDetector()

print("\nRunning predictions...")
results = detector.predict(X)

required_columns = [
    "flow_index",
    "verdict",
    "severity",
    "is_attack",
    "attack_probability",
    "benign_probability",
    "predicted_category",
    "category_confidence",
    "zero_day_anomaly",
    "anomaly_score",
]

missing = [c for c in required_columns if c not in results.columns]
if missing:
    raise AssertionError(f"Missing output columns: {missing}")

assert len(results) == len(X), "Prediction row count mismatch."
assert results["is_attack"].isin([True, False]).all()
assert results["zero_day_anomaly"].isin([True, False]).all()

assert np.isfinite(results["attack_probability"]).all()
assert np.isfinite(results["benign_probability"]).all()
assert np.isfinite(results["anomaly_score"]).all()

assert (
    (results["attack_probability"] >= 0)
    & (results["attack_probability"] <= 100)
).all()

assert np.allclose(
    results["attack_probability"] + results["benign_probability"],
    100,
    atol=0.02
)

# Confirm missing required features are rejected.
try:
    detector.predict(X.drop(columns=[X.columns[0]]))
except ValueError:
    missing_feature_check = "PASSED"
else:
    raise AssertionError("Detector did not reject a missing feature.")

print("\n===== DETECTOR TEST RESULTS =====")
print("Input records:", len(X))
print("Output records:", len(results))
print("Attack predictions:", int(results["is_attack"].sum()))
print("Benign predictions:", int((~results["is_attack"]).sum()))
print("Anomaly flags:", int(results["zero_day_anomaly"].sum()))
print("Missing-feature validation:", missing_feature_check)
print("All probability values valid: YES")
print("All required output fields present: YES")

print("\nSample predictions:")
print(results.head(5).to_string(index=False))

print("\nDETECTOR TEST PASSED")
print("No model or dataset files were modified.")
