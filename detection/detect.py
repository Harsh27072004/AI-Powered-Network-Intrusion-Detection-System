import pandas as pd
import joblib
import os

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/random_forest_binary.pkl"
TEST_FILE = "dataset/processed/prepared/X_test.csv"

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 70)
print("AI-NIDS DETECTION ENGINE")
print("=" * 70)

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# =========================================================
# LOAD SAMPLE NETWORK FLOWS
# =========================================================

print("\nLoading test network flows...")

X = pd.read_csv(TEST_FILE)

print("Total flows available:", len(X))
print("Features:", X.shape[1])

# =========================================================
# SELECT A FEW FLOWS FOR DEMONSTRATION
# =========================================================

sample = X.head(10)

# =========================================================
# PREDICT
# =========================================================

print("\n" + "=" * 70)
print("DETECTION RESULTS")
print("=" * 70)

predictions = model.predict(sample)

probabilities = model.predict_proba(sample)

for i in range(len(sample)):

    prediction = predictions[i]

    benign_probability = probabilities[i][0]
    attack_probability = probabilities[i][1]

    if prediction == 1:
        result = "ATTACK"
        confidence = attack_probability
    else:
        result = "BENIGN"
        confidence = benign_probability

    print("\nFlow:", i + 1)
    print("Prediction:", result)
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"BENIGN probability: {benign_probability * 100:.2f}%")
    print(f"ATTACK probability: {attack_probability * 100:.2f}%")

print("\n" + "=" * 70)
print("DETECTION COMPLETE")
print("=" * 70)