
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "dataset" / "processed" / "binary_nids_dataset.csv"
OUTPUT = ROOT / "dataset" / "processed" / "binary_nids_dataset_v3.csv"

# These are the two verified conflicting feature patterns.
CONFLICT_HASHES = {
    16477926396787665471,
    9754427028844431846,
}

if not SOURCE.exists():
    raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT.name} already exists. "
        "Stopping to protect the existing file."
    )

# Identify feature columns.
columns = pd.read_csv(SOURCE, nrows=0).columns.tolist()
features = [
    col for col in columns
    if col not in ("Label", "Binary_Label")
]

kept_chunks = []
removed_count = 0
total_count = 0

for chunk in pd.read_csv(SOURCE, chunksize=50000, low_memory=False):
    total_count += len(chunk)

    hashes = pd.util.hash_pandas_object(
        chunk[features], index=False
    )

    remove_mask = hashes.isin(CONFLICT_HASHES)
    removed_count += int(remove_mask.sum())

    kept_chunks.append(chunk.loc[~remove_mask].copy())

cleaned = pd.concat(kept_chunks, ignore_index=True)

cleaned.to_csv(OUTPUT, index=False)

print("\nV3 DATASET CREATED")
print("Original rows:", total_count)
print("Removed ambiguous rows:", removed_count)
print("Remaining rows:", len(cleaned))
print("\nRemaining label distribution:")
print(cleaned["Label"].value_counts())
print("\nOutput:", OUTPUT)
print("\nOriginal dataset was not modified.")
