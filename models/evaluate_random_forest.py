import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# SETTINGS
# =========================================================

DATA_DIR = "dataset/processed/prepared"
MODEL_PATH = "models/random_forest_binary.pkl"

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 70)
print("FINAL TEST EVALUATION - BINARY NIDS")
print("=" * 70)

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# =========================================================
# LOAD TEST DATA
# =========================================================

print("\nLoading test data...")

X_test = pd.read_csv(
    os.path.join(DATA_DIR, "X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
).squeeze()

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

# =========================================================
# PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# Probability of ATTACK
y_probability = model.predict_proba(X_test)[:, 1]

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = fp / (fp + tn)

false_negative_rate = fn / (fn + tp)

# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC-AUC            : {roc_auc:.4f}")

print(f"\nFalse Positive Rate: {false_positive_rate:.6f}")
print(f"False Negative Rate: {false_negative_rate:.6f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n              Predicted")
print("              BENIGN  ATTACK")

print(
    f"Actual BENIGN  {tn:7d}  {fp:7d}"
)

print(
    f"Actual ATTACK  {fn:7d}  {tp:7d}"
)

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION COMPLETE")
print("=" * 70)