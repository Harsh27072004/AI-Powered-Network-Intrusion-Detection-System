
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed"
MODEL_PATH = ROOT / "models" / "random_forest_binary_v2.pkl"
MEDIAN_PATH = DATA / "prepared_v2" / "training_medians.csv"

# Reconstruct the exact original test-row positions.
df = pd.read_csv(DATA / "binary_nids_dataset.csv", low_memory=False)

from sklearn.model_selection import train_test_split

X = df.drop(columns=["Label", "Binary_Label"])
y = df["Binary_Label"]

_, X_temp, _, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# The saved original split was verified to match this reconstruction.
test_indices = X_test.index
original_labels = df.loc[test_indices, "Label"].astype(str).str.strip()

features = pd.read_csv(
    DATA / "prepared_v2" / "X_train.csv", nrows=0
).columns.tolist()

X_eval = X_test[features].copy()
X_eval = X_eval.apply(pd.to_numeric, errors="coerce")
X_eval = X_eval.replace([np.inf, -np.inf], np.nan)

medians = pd.read_csv(MEDIAN_PATH, index_col=0)["median"]
X_eval = X_eval.fillna(medians).fillna(0).astype("float32")

model = joblib.load(MODEL_PATH)
attack_index = list(model.classes_).index(1)
scores = model.predict_proba(X_eval)[:, attack_index]

threshold_path = ROOT / "models" / "random_forest_binary_v2_threshold.txt"
threshold = float(threshold_path.read_text(encoding="utf-8").strip())
pred = (scores >= threshold).astype(np.int8)

print("V2 ATTACK-CATEGORY EVALUATION")
print(f"Threshold: {threshold:.4f}")
print(f"Test rows: {len(X_eval):,}")
print("\nCategory-level results:")

rows = []
for label in sorted(original_labels.unique()):
    mask = (original_labels.to_numpy() == label)
    count = int(mask.sum())

    if label == "BENIGN":
        false_alarms = int(pred[mask].sum())
        rate = false_alarms / count if count else 0
        print(
            f"{label}: {count:,} flows; false alarms={false_alarms:,}; "
            f"false-positive rate={rate:.4%}"
        )
    else:
        detected = int(pred[mask].sum())
        recall = detected / count if count else 0
        print(
            f"{label}: {count:,} flows; detected={detected:,}; "
            f"missed={count-detected:,}; recall={recall:.4%}"
        )

    rows.append({
        "Label": label,
        "Total": count,
        "Predicted_Attack": int(pred[mask].sum()),
        "Recall_or_FPR": (
            int(pred[mask].sum()) / count if count else 0
        ),
    })

out = ROOT / "reports" / "v2_category_evaluation.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)

print(f"\nSaved category summary: {out}")
print("No model or dataset files modified.")
