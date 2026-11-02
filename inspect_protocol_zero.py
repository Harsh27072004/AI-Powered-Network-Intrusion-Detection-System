
import pandas as pd
from detection.unified_detector import UnifiedNIDSDetector

raw = pd.read_csv(
    "tools/CICFlowMeter_corrected/data/daily/2026-11-01_Flow.csv"
)

features = pd.read_csv("reports/live_capture_03_v3.csv")

detector = UnifiedNIDSDetector()
pred = detector.predict(features)

result = pd.DataFrame({
    "Protocol": raw["Protocol"],
    "Src IP": raw["Src IP"],
    "Src Port": raw["Src Port"],
    "Dst IP": raw["Dst IP"],
    "Dst Port": raw["Dst Port"],
    "Flow Duration": features["Flow Duration"],
    "Fwd Packets": features["Total Fwd Packets"],
    "Bwd Packets": features["Total Backward Packets"],
    "Verdict": pred["verdict"],
    "Probability": pred["attack_probability"]
})

zero = result[result["Protocol"] == 0]

print("\nPROTOCOL 0 SUMMARY")
print("Total flows:", len(zero))
print("\nDestination ports:")
print(zero["Dst Port"].value_counts())

print("\nVERDICTS:")
print(zero["Verdict"].value_counts())

print("\nFIRST 15 PROTOCOL-0 FLOWS:")
print(zero.head(15).to_string(index=False))
