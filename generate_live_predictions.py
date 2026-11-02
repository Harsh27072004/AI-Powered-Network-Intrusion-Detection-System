import pandas as pd
from detection.unified_detector import UnifiedNIDSDetector

# Load the latest live capture
input_file = "reports/live_capture_03_v3.csv"

df = pd.read_csv(input_file)

# Initialize detector
detector = UnifiedNIDSDetector()

# Generate predictions
results = detector.predict(df)

# Save predictions
output_file = "reports/live_capture_03_predictions.csv"
results.to_csv(output_file, index=False)

# Display summary
print("\nPrediction Summary:")
print(results["verdict"].value_counts())

print("\nTotal flows:", len(results))
print("Saved to:", output_file)