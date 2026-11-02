
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.unified_detector import UnifiedNIDSDetector

INPUT = ROOT / "reports" / "live_flows_v1_backup.csv"
OUTPUT = ROOT / "reports" / "live_flow_detection_results.csv"

print("Loading captured network flows...")

flows = pd.read_csv(INPUT)

print("Flows loaded:", len(flows))
print("Features:", len(flows.columns))

detector = UnifiedNIDSDetector()

print("\nDetecting network flows...")

predictions = detector.predict(flows)

results = pd.concat(
    [flows.reset_index(drop=True),
     predictions.reset_index(drop=True)],
    axis=1
)

results.to_csv(OUTPUT, index=False)

print("\n===== LIVE FLOW DETECTION RESULTS =====")
print(predictions["verdict"].value_counts().to_string())

print("\nTotal flows:", len(predictions))
print("Attack predictions:", int(predictions["is_attack"].sum()))
print("Benign predictions:", int((~predictions["is_attack"]).sum()))

print("\nResults saved:", OUTPUT)
