import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.util import hash_pandas_object

# ==============================================================================
# LEAK-FREE DATA PREPARATION & STRATIFIED SPLITTER
# AI-NIDS Professional Pipeline
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "processed", "binary_nids_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "processed", "prepared")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 17 Features to remove (constant or redundant) to arrive at the 61 standard features
FEATURES_TO_REMOVE = [
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
    "ECE Flag Count"
]

# Attack family mapping for Multi-Class classification
ATTACK_MAPPING = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Bot": "Botnet",
    "Infiltration": "Botnet"
}

def clean_label(val):
    if not isinstance(val, str):
        return "BENIGN"
    val = val.strip()
    if "Web Attack" in val:
        return "WebAttack"
    return ATTACK_MAPPING.get(val, "ATTACK" if val != "BENIGN" else "BENIGN")

print("=" * 75)
print("PHASE 1: LEAK-FREE DATA PREPARATION & MULTI-CLASS LABELING")
print("=" * 75)

print("\n[1/6] Loading source dataset...")
df = pd.read_csv(INPUT_FILE, encoding="latin-1")
print(f"Loaded {len(df):,} total records with {len(df.columns)} columns.")

# Handle label columns
labels_raw = df["Label"]
binary_target = df["Binary_Label"].astype(int)

# Create clean multi-class labels
multiclass_target = labels_raw.apply(clean_label)

print("\nMulti-Class Target Distribution:")
print(multiclass_target.value_counts().to_string())

# Separate feature matrix
X = df.drop(columns=["Label", "Binary_Label"], errors="ignore")

# Strip whitespace from column names if present
X.columns = [c.strip() for c in X.columns]

# Drop redundant / constant features to match the exact 61 features
cols_to_drop = [c for c in FEATURES_TO_REMOVE if c in X.columns]
X = X.drop(columns=cols_to_drop)

print(f"\n[2/6] Feature selection: Reduced to {X.shape[1]} standard features.")
assert X.shape[1] == 61, f"Expected 61 features, got {X.shape[1]}"

# Replace infinities and handle NaNs
print("[3/6] Cleaning infinite and missing values...")
X = X.replace([np.inf, -np.inf], np.nan)
nan_count = X.isnull().sum().sum()
if nan_count > 0:
    print(f"Filling {nan_count:,} NaN values with column medians...")
    X = X.fillna(X.median())

# ==============================================================================
# RESOLVE CONFLICTING AND DUPLICATE FLOWS
# ==============================================================================
print("\n[4/6] Analyzing duplicate and conflicting flow vectors...")
flow_hashes = hash_pandas_object(X, index=False)

# Check for duplicate feature vectors
total_rows = len(X)
unique_hashes = len(flow_hashes.unique())
print(f"Total rows: {total_rows:,} | Unique flow patterns: {unique_hashes:,}")
print(f"Duplicate rows detected: {total_rows - unique_hashes:,}")

# Identify conflicting feature vectors (same features, different binary labels)
combined = pd.DataFrame({
    "hash": flow_hashes,
    "binary": binary_target,
    "multi": multiclass_target
})

# Find hashes with more than one unique binary label
hash_label_counts = combined.groupby("hash")["binary"].nunique()
conflicting_hashes = set(hash_label_counts[hash_label_counts > 1].index)
print(f"Found {len(conflicting_hashes):,} feature patterns with conflicting labels.")

# Filter out ambiguous/conflicting rows
valid_mask = ~flow_hashes.isin(conflicting_hashes)
print(f"Retaining {valid_mask.sum():,} clean rows (removed {len(X) - valid_mask.sum():,} ambiguous rows).")

X_clean = X.loc[valid_mask].copy()
y_bin_clean = binary_target.loc[valid_mask].copy()
y_mul_clean = multiclass_target.loc[valid_mask].copy()

# Deduplicate identical feature vectors to avoid memorization and leakage
print("\nDeduplicating exact feature rows to ensure zero memorization...")
dup_mask = ~flow_hashes.loc[valid_mask].duplicated(keep="first")
X_dedup = X_clean.loc[dup_mask].copy()
y_bin_dedup = y_bin_clean.loc[dup_mask].copy()
y_mul_dedup = y_mul_clean.loc[dup_mask].copy()

print(f"Final clean unique dataset: {len(X_dedup):,} rows.")
print("\nFinal Clean Multi-Class Distribution:")
print(y_mul_dedup.value_counts().to_string())

# ==============================================================================
# STRATIFIED LEAK-FREE SPLIT (70% Train, 15% Validation, 15% Test)
# ==============================================================================
print("\n[5/6] Splitting dataset (70% Train, 15% Val, 15% Test)...")
X_train, X_temp, y_bin_train, y_bin_temp, y_mul_train, y_mul_temp = train_test_split(
    X_dedup,
    y_bin_dedup,
    y_mul_dedup,
    test_size=0.30,
    random_state=42,
    stratify=y_mul_dedup
)

X_val, X_test, y_bin_val, y_bin_test, y_mul_val, y_mul_test = train_test_split(
    X_temp,
    y_bin_temp,
    y_mul_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_mul_temp
)

print(f"Train set:      {X_train.shape[0]:,} rows")
print(f"Validation set: {X_val.shape[0]:,} rows")
print(f"Test set:       {X_test.shape[0]:,} rows")

# ==============================================================================
# AUDIT: VERIFY ZERO LEAKAGE BETWEEN SPLITS
# ==============================================================================
print("\n[6/6] Auditing splits for feature overlap...")
h_train = set(hash_pandas_object(X_train, index=False).tolist())
h_val = set(hash_pandas_object(X_val, index=False).tolist())
h_test = set(hash_pandas_object(X_test, index=False).tolist())

train_test_overlap = len(h_train.intersection(h_test))
train_val_overlap = len(h_train.intersection(h_val))
val_test_overlap = len(h_val.intersection(h_test))

print(f"Train <-> Test feature overlap: {train_test_overlap} rows (Target: 0)")
print(f"Train <-> Val  feature overlap: {train_val_overlap} rows (Target: 0)")
print(f"Val   <-> Test feature overlap: {val_test_overlap} rows (Target: 0)")

assert train_test_overlap == 0, "ERROR: Data leakage detected between train and test!"
print(">>> DATA INTEGRITY VERIFIED: ZERO DATA LEAKAGE! <<<")

# ==============================================================================
# SAVE CLEAN PREPARED DATASETS
# ==============================================================================
print("\nSaving clean prepared datasets to:", OUTPUT_DIR)

X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_val.to_csv(os.path.join(OUTPUT_DIR, "X_val.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)

pd.DataFrame({"label": y_bin_train}).to_csv(os.path.join(OUTPUT_DIR, "y_train_binary.csv"), index=False)
pd.DataFrame({"label": y_bin_val}).to_csv(os.path.join(OUTPUT_DIR, "y_val_binary.csv"), index=False)
pd.DataFrame({"label": y_bin_test}).to_csv(os.path.join(OUTPUT_DIR, "y_test_binary.csv"), index=False)

pd.DataFrame({"category": y_mul_train}).to_csv(os.path.join(OUTPUT_DIR, "y_train_multiclass.csv"), index=False)
pd.DataFrame({"category": y_mul_val}).to_csv(os.path.join(OUTPUT_DIR, "y_val_multiclass.csv"), index=False)
pd.DataFrame({"category": y_mul_test}).to_csv(os.path.join(OUTPUT_DIR, "y_test_multiclass.csv"), index=False)

print("\nSUCCESS: All leak-free datasets successfully saved!")
