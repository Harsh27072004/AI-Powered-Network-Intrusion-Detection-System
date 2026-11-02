
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed" / "prepared_v3"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODEL_PATH = MODELS / "random_forest_binary_v3.pkl"
THRESHOLD_PATH = MODELS / "random_forest_binary_v3_threshold.txt"
REPORT_PATH = REPORTS / "random_forest_binary_v3_report.txt"

for folder in (DATA, MODELS, REPORTS):
    if not folder.exists():
        raise FileNotFoundError(f"Required folder missing: {folder}")

if MODEL_PATH.exists() or THRESHOLD_PATH.exists() or REPORT_PATH.exists():
    raise FileExistsError(
        "A V3 model/report file already exists. "
        "Stopping to protect existing work."
    )

print("Loading V3 training data...")

X_train = pd.read_csv(DATA / "X_train.csv")
X_val = pd.read_csv(DATA / "X_validation.csv")
X_test = pd.read_csv(DATA / "X_test.csv")

y_train = pd.read_csv(DATA / "y_train.csv")["Binary_Label"]
y_val = pd.read_csv(DATA / "y_validation.csv")["Binary_Label"]
y_test = pd.read_csv(DATA / "y_test.csv")["Binary_Label"]

if not (list(X_train.columns) == list(X_val.columns) == list(X_test.columns)):
    raise ValueError("Feature columns/order do not match.")

if not (len(X_train) == len(y_train)
        and len(X_val) == len(y_val)
        and len(X_test) == len(y_test)):
    raise ValueError("Feature and label row counts do not match.")

if any(frame.isna().any().any() for frame in (X_train, X_val, X_test)):
    raise ValueError("Missing values remain in prepared data.")

if not all(np.isfinite(frame.to_numpy()).all()
           for frame in (X_train, X_val, X_test)):
    raise ValueError("Infinite values remain in prepared data.")

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Features:", X_train.shape[1])

print("\nTraining Random Forest...")
start = time.time()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
    max_features="sqrt",
)

model.fit(X_train, y_train)

print(f"Training completed in {(time.time() - start) / 60:.1f} minutes.")

# Use validation data only to select the operating threshold.
val_probabilities = model.predict_proba(X_val)[:, 1]

threshold_results = []
for threshold in np.arange(0.05, 1.00, 0.01):
    predictions = (val_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    if fpr <= 0.01:
        threshold_results.append((recall, threshold, fpr))

if not threshold_results:
    raise RuntimeError("No validation threshold satisfies FPR <= 1%.")

# Highest recall under the validation FPR constraint.
best_recall, best_threshold, best_fpr = max(
    threshold_results, key=lambda item: (item[0], item[1])
)

print("\nSelected threshold:", round(float(best_threshold), 4))
print("Validation recall:", round(float(best_recall), 6))
print("Validation FPR:", round(float(best_fpr), 6))

# Evaluate on untouched test data only after threshold selection.
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

test_fpr = fp / (fp + tn) if (fp + tn) else 0

metrics = {
    "threshold": float(best_threshold),
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
    "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
    "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
    "false_positive_rate": float(test_fpr),
    "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "average_precision": float(average_precision_score(y_test, test_probabilities)),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}

report = []
report.append("AI-NIDS V3 RANDOM FOREST EVALUATION")
report.append("=" * 45)
report.append("Split: Group-based V3 split")
report.append("Threshold selected using validation data only")
report.append("Validation FPR constraint: <= 1%")
report.append("")
report.append("TEST SET METRICS")
for key, value in metrics.items():
    report.append(f"{key}: {value}")
report.append("")
report.append("Classification report:")
report.append(
    classification_report(
        y_test, test_predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)
report.append("Confusion matrix [BENIGN, ATTACK]:")
report.append(str(confusion_matrix(y_test, test_predictions, labels=[0, 1])))

# Save only to new V3 paths.
joblib.dump(model, MODEL_PATH)
THRESHOLD_PATH.write_text(str(float(best_threshold)), encoding="utf-8")
REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

# Save feature importance separately.
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv(
    REPORTS / "random_forest_binary_v3_feature_importance.csv",
    index=False
)

print("\n===== V3 MODEL TRAINING COMPLETE =====")
print("Test accuracy:", round(metrics["accuracy"], 6))
print("Test precision:", round(metrics["precision"], 6))
print("Test recall:", round(metrics["recall"], 6))
print("Test F1:", round(metrics["f1"], 6))
print("Test FPR:", round(metrics["false_positive_rate"], 6))
print("ROC AUC:", round(metrics["roc_auc"], 6))
print("Average precision:", round(metrics["average_precision"], 6))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))
print("\nModel saved:", MODEL_PATH)
print("Report saved:", REPORT_PATH)
print("Previous models were not modified.")
