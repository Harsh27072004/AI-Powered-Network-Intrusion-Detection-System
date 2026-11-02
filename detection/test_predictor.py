import pandas as pd

from predictor import predict_flow


# =========================================================
# LOAD ONE TEST FLOW
# =========================================================

TEST_FILE = "dataset/processed/prepared/X_test.csv"

print("=" * 70)
print("TESTING REUSABLE AI-NIDS PREDICTOR")
print("=" * 70)

X_test = pd.read_csv(TEST_FILE)

# Take one network flow
flow = X_test.iloc[0].to_dict()

print("\nTesting one network flow...")

# =========================================================
# PREDICT
# =========================================================

result = predict_flow(flow)

# =========================================================
# DISPLAY RESULT
# =========================================================

print("\n" + "=" * 70)
print("DETECTION RESULT")
print("=" * 70)

print("Prediction:", result["prediction"])

print(
    f"Confidence: "
    f"{result['confidence'] * 100:.2f}%"
)

print(
    f"BENIGN probability: "
    f"{result['benign_probability'] * 100:.2f}%"
)

print(
    f"ATTACK probability: "
    f"{result['attack_probability'] * 100:.2f}%"
)

print("\n" + "=" * 70)
print("PREDICTOR TEST COMPLETE")
print("=" * 70)