
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "dataset" / "processed" / "splits"
OUT = ROOT / "dataset" / "processed" / "prepared_v2"
OUT.mkdir(parents=True, exist_ok=True)

DROP = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "CWE Flag Count", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Bwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Bytes", "Avg Fwd Segment Size",
    "Fwd Header Length.1", "ECE Flag Count",
]

names = ["train", "validation", "test"]
X_sets, y_sets = {}, {}

# Load the original splits and verify row alignment first.
for name in names:
    X_path = SPLITS / f"X_{name}.csv"
    y_path = SPLITS / f"y_{name}.csv"

    X = pd.read_csv(X_path, low_memory=False)
    y = pd.read_csv(y_path, low_memory=False)

    if len(X) != len(y):
        raise ValueError(
            f"{name}: X has {len(X)} rows but y has {len(y)}"
        )

    X = X.drop(columns=[c for c in DROP if c in X.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    X_sets[name] = X
    y_sets[name] = y

# Learn imputations from training data only.
medians = X_sets["train"].median().fillna(0)
expected = X_sets["train"].columns.tolist()

for name in names:
    X = X_sets[name].reindex(columns=expected)
    X = X.fillna(medians).fillna(0)

    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"{name}: invalid values remain")

    if len(X) != len(y_sets[name]):
        raise ValueError(f"{name}: row alignment was lost")

    X.to_csv(OUT / f"X_{name}.csv", index=False)
    y_sets[name].to_csv(OUT / f"y_{name}.csv", index=False)

    print(
        f"{name}: {len(X):,} rows, {len(X.columns)} features, "
        f"{len(y_sets[name]):,} labels"
    )

medians.rename("median").to_csv(OUT / "training_medians.csv")

print("\nFeature preparation v2 complete.")
print("Output:", OUT)
print("Original files and models were not modified.")
