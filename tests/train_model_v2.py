
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed" / "prepared_v2"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def load_split(name):
    X = pd.read_csv(DATA / f"X_{name}.csv", low_memory=False)
    y_df = pd.read_csv(DATA / f"y_{name}.csv", low_memory=False)

    if len(y_df.columns) != 1:
        raise ValueError(f"{name}: expected one label column")

    y = pd.to_numeric(y_df.iloc[:, 0], errors="raise")

    if len(X) != len(y):
        raise ValueError(f"{name}: feature/label row mismatch")

    if set(y.unique()) != {0, 1}:
        raise ValueError(f"{name}: expected binary labels 0 and 1")

    return X, y.to_numpy(dtype=np.int8)

X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")

if list(X_train.columns) != list(X_val.columns):
    raise ValueError("Training and validation feature columns differ")
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and test feature columns differ")

print("TRAINING DATA")
print("Train:", X_train.shape, "labels:", np.bincount(y_train))
print("Validation:", X_val.shape, "labels:", np.bincount(y_val))
print("Test:", X_test.shape, "labels:", np.bincount(y_test))

print("\nTraining a separate Random Forest...")
start = time.time()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
model.fit(X_train, y_train)

print(f"Training completed in {(time.time() - start) / 60:.1f} minutes")

attack_index = list(model.classes_).index(1)

def metrics(y_true, scores, threshold=0.5):
    pred = (scores >= threshold).astype(np.int8)

    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    fpr = fp / max(fp + tn, 1)

    return {
        "threshold": threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }

print("\nEvaluating validation split...")
val_scores = model.predict_proba(X_val)[:, attack_index]

# Choose a threshold using validation data only.
# Among thresholds with validation FPR <= 1%, maximize recall.
thresholds = np.linspace(0.05, 0.95, 91)
eligible = []

for threshold in thresholds:
    result = metrics(y_val, val_scores, float(threshold))
    if result["false_positive_rate"] <= 0.01:
        eligible.append(result)

if eligible:
    selected = max(
        eligible,
        key=lambda r: (r["recall"], r["precision"], r["threshold"])
    )
    selected_threshold = selected["threshold"]
else:
    selected_threshold = 0.5

print(f"Validation-selected threshold: {selected_threshold:.2f}")
print("Selection rule: highest validation recall with FPR <= 1%")

print("\nValidation metrics at threshold 0.50:")
val_default = metrics(y_val, val_scores, 0.5)
print(val_default)

print("\nValidation metrics at selected threshold:")
val_selected = metrics(y_val, val_scores, selected_threshold)
print(val_selected)

# Test split is evaluated only after threshold selection.
print("\nEvaluating random test split...")
test_scores = model.predict_proba(X_test)[:, attack_index]
test_default = metrics(y_test, test_scores, 0.5)
test_selected = metrics(y_test, test_scores, selected_threshold)

print("\nRandom test metrics at threshold 0.50:")
print(test_default)
print("\nRandom test metrics at validation-selected threshold:")
print(test_selected)

model_path = MODELS / "random_forest_binary_v2.pkl"
threshold_path = MODELS / "random_forest_binary_v2_threshold.txt"
report_path = REPORTS / "random_forest_binary_v2_report.txt"

joblib.dump(model, model_path)
threshold_path.write_text(
    f"{selected_threshold:.4f}\n", encoding="utf-8"
)

report = [
    "AI-NIDS EXPERIMENTAL MODEL V2",
    "Training data: prepared_v2",
    "Threshold selection: validation only; maximize recall with FPR <= 1%",
    f"Selected threshold: {selected_threshold:.4f}",
    f"Validation at 0.50: {val_default}",
    f"Validation at selected threshold: {val_selected}",
    f"Random test at 0.50: {test_default}",
    f"Random test at selected threshold: {test_selected}",
    "",
    "Caution: the random split may share source-specific patterns.",
    "Use the separate source-held-out experiment to assess generalization.",
]

report_path.write_text("\n".join(report), encoding="utf-8")

print("\nSaved new experimental artifacts:")
print(model_path)
print(threshold_path)
print(report_path)
print("Existing baseline model and splits were not overwritten.")
