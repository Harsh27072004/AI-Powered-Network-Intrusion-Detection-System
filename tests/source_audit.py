
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"

files = sorted(DATA_DIR.glob("*_cleaned.csv"))

if not files:
    raise FileNotFoundError(f"No cleaned CSV files found in {DATA_DIR}")

print("=" * 75)
print("AI-NIDS SOURCE DATA AUDIT")
print("=" * 75)

reference_features = None
feature_mismatches = []
total_rows = 0
total_benign = 0
total_attack = 0

for path in files:
    df = pd.read_csv(path, low_memory=False)

    if "Label" not in df.columns:
        print(f"\nWARNING: {path.name} has no Label column")
        continue

    features = set(df.columns) - {"Label"}
    if reference_features is None:
        reference_features = features
    elif features != reference_features:
        feature_mismatches.append(path.name)

    labels = df["Label"].astype(str).str.strip()
    benign = int((labels == "BENIGN").sum())
    attack = len(df) - benign

    total_rows += len(df)
    total_benign += benign
    total_attack += attack

    print(f"\nFile: {path.name}")
    print(f"Rows: {len(df):,}")
    print(f"Benign: {benign:,}")
    print(f"Non-benign: {attack:,}")
    print("Labels:", labels.value_counts().to_dict())

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"Files inspected: {len(files)}")
print(f"Total rows: {total_rows:,}")
print(f"Total benign rows: {total_benign:,}")
print(f"Total non-benign rows: {total_attack:,}")
print(f"Reference feature count excluding Label: {len(reference_features or [])}")

if feature_mismatches:
    print("Files with different feature columns:", feature_mismatches)
else:
    print("Feature column names: consistent across all files")

print("\nAudit complete. No datasets or models were modified.")
