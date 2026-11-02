
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.unified_detector import UnifiedNIDSDetector

INPUT = ROOT / "reports" / "pcap_test_predictions.csv"
OUTPUT = ROOT / "reports" / "pcap_test_predictions_v2.csv"

print("Loading existing PCAP test data...")

df = pd.read_csv(INPUT)

# Remove previous prediction columns.
prediction_columns = [
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

flows = df.drop(
    columns=prediction_columns,
    errors="ignore"
)

print("Input flows:", len(flows))

detector = UnifiedNIDSDetector()

print("\nRunning predictions...")

predictions = detector.predict(flows)

results = pd.concat(
    [flows.reset_index(drop=True),
     predictions.reset_index(drop=True)],
    axis=1
)

results.to_csv(OUTPUT, index=False)

print("\n===== UPDATED DETECTION RESULTS =====")
print(predictions["verdict"].value_counts().to_string())

print("\nAttack predictions:",
      int(predictions["is_attack"].sum()))

print("Benign predictions:",
      int((~predictions["is_attack"]).sum()))

print("\nSaved:", OUTPUT)
