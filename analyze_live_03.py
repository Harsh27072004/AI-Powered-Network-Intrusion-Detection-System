
import pandas as pd
from detection.unified_detector import UnifiedNIDSDetector

features = pd.read_csv("reports/live_capture_03_v3.csv")
raw = pd.read_csv(
    "tools/CICFlowMeter_corrected/data/daily/2026-11-01_Flow.csv"
)

detector = UnifiedNIDSDetector()
pred = detector.predict(features)

result = pd.DataFrame({
    "port": features["Destination Port"],
    "protocol": raw["Protocol"],
    "verdict": pred["verdict"],
    "probability": pred["attack_probability"]
})

print("\nRESULTS BY DESTINATION PORT:\n")

summary = result.groupby("port").agg(
    flows=("verdict", "size"),
    attacks=("verdict", lambda x: x.str.startswith("ATTACK").sum()),
    mean_probability=("probability", "mean")
)

print(summary.sort_values("flows", ascending=False).head(20).to_string())

print("\nRESULTS BY PROTOCOL:\n")

print(result.groupby("protocol").agg(
    flows=("verdict", "size"),
    attacks=("verdict", lambda x: x.str.startswith("ATTACK").sum()),
    mean_probability=("probability", "mean")
).to_string())
