
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_cleaned.csv"
MAX_PER_CLASS_PER_FILE = 40000
RANDOM_STATE = 42

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Reuse the exact feature names and order from the existing training data.
feature_cols = pd.read_csv(
    DATA_DIR / "prepared" / "X_train.csv", nrows=0
).columns.tolist()

if len(feature_cols) != 61:
    raise ValueError(f"Expected 61 features, found {len(feature_cols)}")

source_files = sorted(DATA_DIR.glob("*_cleaned.csv"))
test_paths = [p for p in source_files if p.name == TARGET_FILE]
train_paths = [p for p in source_files if p.name != TARGET_FILE]

if len(test_paths) != 1 or len(train_paths) != 7:
    raise RuntimeError(
        f"Expected 8 source files, found {len(source_files)}; "
        f"Tuesday test files: {len(test_paths)}"
    )

def load_source(path):
    df = pd.read_csv(
        path, usecols=feature_cols + ["Label"], low_memory=False
    )
    df = df[feature_cols + ["Label"]]

    # Convert all features to numeric; invalid entries become missing.
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df["Label"].astype(str).str.strip().ne("BENIGN").astype("int8")

    return X, y

print("Loading training sources and sampling each file...")
train_X_parts = []
train_y_parts = []
train_counts = []

for path in train_paths:
    X, y = load_source(path)
    sampled_indices = []

    for label in (0, 1):
        indices = y.index[y == label].to_numpy()

        if len(indices) > MAX_PER_CLASS_PER_FILE:
            rng = np.random.default_rng(
                RANDOM_STATE + len(train_counts) * 10 + label
            )
            indices = rng.choice(
                indices, size=MAX_PER_CLASS_PER_FILE, replace=False
            )

        sampled_indices.extend(indices.tolist())

    sampled_indices = np.array(sampled_indices, dtype=int)
    X_sample = X.loc[sampled_indices].copy()
    y_sample = y.loc[sampled_indices].copy()

    train_X_parts.append(X_sample)
    train_y_parts.append(y_sample)

    train_counts.append(
        f"{path.name}: selected={len(y_sample):,}, "
        f"benign={(y_sample == 0).sum():,}, "
        f"attack={(y_sample == 1).sum():,}"
    )
    print(train_counts[-1])

X_train = pd.concat(train_X_parts, ignore_index=True)
y_train = pd.concat(train_y_parts, ignore_index=True)

# Learn imputation values from training data only.
medians = X_train.median(axis=0).fillna(0)
X_train = X_train.fillna(medians).astype("float32")

print(f"\nTraining rows: {len(X_train):,}")
print("Training class counts:", y_train.value_counts().to_dict())

print("\nLoading the held-out Tuesday source...")
X_test, y_test = load_source(test_paths[0])
X_test = X_test.fillna(medians).astype("float32")

print(f"Test rows: {len(X_test):,}")
print("Test class counts:", y_test.value_counts().to_dict())

print("\nTraining experimental Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
model.fit(X_train, y_train)

print("Evaluating on Tuesday...")
pred = model.predict(X_test)

y_true = y_test.to_numpy()
pred = np.asarray(pred)

tn = int(((y_true == 0) & (pred == 0)).sum())
fp = int(((y_true == 0) & (pred == 1)).sum())
fn = int(((y_true == 1) & (pred == 0)).sum())
tp = int(((y_true == 1) & (pred == 1)).sum())

accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-15)

report = [
    "AI-NIDS SOURCE-HOLDOUT EVALUATION",
    f"Test source: {TARGET_FILE}",
    "Training sources:",
    *[f"  {s}" for s in train_counts],
    f"Training rows: {len(X_train)}",
    f"Training labels: {y_train.value_counts().to_dict()}",
    f"Test rows: {len(X_test)}",
    f"Test labels: {y_test.value_counts().to_dict()}",
    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
    f"Accuracy:  {accuracy:.6%}",
    f"Precision: {precision:.6%}",
    f"Recall:    {recall:.6%}",
    f"F1-score:  {f1:.6%}",
    "",
    "LIMITATION: This is a source-file holdout, not proof of",
    "performance on all unseen real-world traffic. Duplicate or",
    "related flows may still occur across source files.",
]

report_text = "\n".join(report)
print("\n" + report_text)

model_path = MODEL_DIR / "source_holdout_friday_ddos_rf.pkl"
medians_path = MODEL_DIR / "source_holdout_friday_ddos_medians.csv"
report_path = REPORT_DIR / "source_holdout_friday_ddos_report.txt"

joblib.dump(model, model_path)
medians.rename("median").to_csv(medians_path, header=True)
report_path.write_text(report_text, encoding="utf-8")

print("\nSaved separate experimental artifacts:")
print(model_path)
print(medians_path)
print(report_path)
print("\nExisting baseline model and split files were not modified.")
