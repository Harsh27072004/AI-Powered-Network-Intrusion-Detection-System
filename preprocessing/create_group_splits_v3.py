
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "dataset" / "processed" / "binary_nids_dataset_v3.csv"
OUTPUT = ROOT / "dataset" / "processed" / "prepared_v3"

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise FileExistsError(
        "prepared_v3 already contains files. "
        "Stopping to protect existing data."
    )

OUTPUT.mkdir(parents=True, exist_ok=True)

DROP_FEATURES = [
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Avg Fwd Segment Size",
    "Fwd Header Length.1",
    "ECE Flag Count",
]

print("Loading V3 dataset...")
df = pd.read_csv(SOURCE, low_memory=False)

if "Binary_Label" not in df.columns:
    raise ValueError("Binary_Label column is missing.")

if df["Binary_Label"].isna().any():
    raise ValueError("Binary_Label contains missing values.")

y = df["Binary_Label"].astype("int8")

if set(y.unique()) != {0, 1}:
    raise ValueError("Unexpected binary label values.")

# Generate a group ID for each feature pattern.
feature_columns = [
    c for c in df.columns
    if c not in ("Label", "Binary_Label")
]

groups = pd.util.hash_pandas_object(
    df[feature_columns], index=False
).to_numpy()

# Prepare features using the V2 feature-selection rules.
X = df.drop(columns=["Label", "Binary_Label"])

missing_drop = [c for c in DROP_FEATURES if c not in X.columns]
if missing_drop:
    raise ValueError(f"Expected V2 columns missing: {missing_drop}")

X = X.drop(columns=DROP_FEATURES)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

print("Rows:", len(X))
print("Features:", X.shape[1])
print("Unique groups:", len(np.unique(groups)))

# First split: approximately 20% test.
splitter_1 = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

remaining_idx, test_idx = next(
    splitter_1.split(X, y, groups)
)

# Second split: approximately 20% of remaining data for validation.
splitter_2 = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=43
)

train_rel, val_rel = next(
    splitter_2.split(
        X.iloc[remaining_idx],
        y.iloc[remaining_idx],
        groups[remaining_idx]
    )
)

train_idx = remaining_idx[train_rel]
val_idx = remaining_idx[val_rel]

# Verify group separation.
train_groups = set(groups[train_idx])
val_groups = set(groups[val_idx])
test_groups = set(groups[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# Fit imputation values on training data only.
medians = X.iloc[train_idx].median().fillna(0)

X_train = X.iloc[train_idx].fillna(medians).fillna(0)
X_val = X.iloc[val_idx].fillna(medians).fillna(0)
X_test = X.iloc[test_idx].fillna(medians).fillna(0)

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]
y_test = y.iloc[test_idx]

# Save separate V3 outputs.
X_train.to_csv(OUTPUT / "X_train.csv", index=False)
X_val.to_csv(OUTPUT / "X_validation.csv", index=False)
X_test.to_csv(OUTPUT / "X_test.csv", index=False)

y_train.to_frame("Binary_Label").to_csv(
    OUTPUT / "y_train.csv", index=False
)
y_val.to_frame("Binary_Label").to_csv(
    OUTPUT / "y_validation.csv", index=False
)
y_test.to_frame("Binary_Label").to_csv(
    OUTPUT / "y_test.csv", index=False
)

medians.to_csv(OUTPUT / "training_medians.csv", header=["median"])

print("\n===== V3 GROUP SPLIT COMPLETE =====")
for name, indices, labels in [
    ("TRAIN", train_idx, y_train),
    ("VALIDATION", val_idx, y_val),
    ("TEST", test_idx, y_test),
]:
    print(f"\n{name}")
    print("Rows:", len(indices))
    print("Benign:", int((labels == 0).sum()))
    print("Attack:", int((labels == 1).sum()))

print("\nFeature count:", X_train.shape[1])
print("Train/validation group overlap:", len(train_groups & val_groups))
print("Train/test group overlap:", len(train_groups & test_groups))
print("Validation/test group overlap:", len(val_groups & test_groups))
print("\nOutput folder:", OUTPUT)
print("Original dataset and previous models were not modified.")
