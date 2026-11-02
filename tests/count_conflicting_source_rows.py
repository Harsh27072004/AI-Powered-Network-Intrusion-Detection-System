
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed"
V2 = DATA / "prepared_v2"

FILES = [
    "Monday-WorkingHours.pcap_ISCX_cleaned.csv",
    "Tuesday-WorkingHours.pcap_ISCX_cleaned.csv",
    "Wednesday-workingHours.pcap_ISCX_cleaned.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_cleaned.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_cleaned.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX_cleaned.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_cleaned.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_cleaned.csv",
]

features = pd.read_csv(V2 / "X_train.csv", nrows=0).columns.tolist()
medians = pd.read_csv(
    V2 / "training_medians.csv", index_col=0
)["median"].reindex(features).fillna(0)

# Recover the two conflicting feature vectors.
seen = {}

for split in ["train", "validation", "test"]:
    X = pd.read_csv(V2 / f"X_{split}.csv", low_memory=False)
    y = pd.read_csv(V2 / f"y_{split}.csv").iloc[:, 0].astype(int)

    for pos, h in enumerate(
        pd.util.hash_pandas_object(X, index=False)
    ):
        h = int(h)
        values = X.iloc[pos].to_numpy(dtype=np.float64)
        label = int(y.iloc[pos])

        if h not in seen:
            seen[h] = [values, {label}]
        elif np.array_equal(seen[h][0], values):
            seen[h][1].add(label)

targets = [
    values for values, labels in seen.values()
    if len(labels) > 1
]

print("Conflicting patterns:", len(targets))

for filename in FILES:
    counts = Counter()

    for chunk in pd.read_csv(
        DATA / filename, chunksize=50000, low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()

        X = chunk[features].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(medians).fillna(0)

        values = X.to_numpy(dtype=np.float64)

        for i, target in enumerate(targets):
            matches = np.all(values == target, axis=1)

            for label, count in chunk.loc[matches, "Label"].value_counts().items():
                counts[(i + 1, str(label))] += int(count)

    print(f"\n{filename}")
    for (pattern, label), count in sorted(counts.items()):
        print(f"Pattern {pattern} | {label}: {count:,}")

print("\nCounting complete. No files modified.")
