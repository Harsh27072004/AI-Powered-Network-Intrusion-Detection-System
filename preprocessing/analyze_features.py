import pandas as pd
import numpy as np

FILE = "dataset/processed/splits/X_train.csv"

print("=" * 70)
print("ADVANCED FEATURE ANALYSIS")
print("=" * 70)

X = pd.read_csv(FILE)

print("\nDataset shape:", X.shape)

# =========================================================
# 1. DUPLICATE COLUMNS
# =========================================================

print("\n" + "=" * 70)
print("DUPLICATE COLUMNS")
print("=" * 70)

duplicate_columns = []

for i in range(len(X.columns)):
    for j in range(i + 1, len(X.columns)):

        col1 = X.columns[i]
        col2 = X.columns[j]

        if X[col1].equals(X[col2]):
            duplicate_columns.append((col1, col2))

if duplicate_columns:
    for pair in duplicate_columns:
        print(pair)
else:
    print("No exact duplicate columns found.")

# =========================================================
# 2. FEATURES WITH NEGATIVE VALUES
# =========================================================

print("\n" + "=" * 70)
print("FEATURES WITH NEGATIVE VALUES")
print("=" * 70)

for column in X.columns:

    minimum = X[column].min()

    if minimum < 0:
        negative_count = (X[column] < 0).sum()

        print(
            f"{column}: "
            f"minimum={minimum}, "
            f"negative_values={negative_count}"
        )

# =========================================================
# 3. UNIQUE VALUE COUNTS
# =========================================================

print("\n" + "=" * 70)
print("LOW-CARDINALITY FEATURES")
print("=" * 70)

for column in X.columns:

    unique_count = X[column].nunique()

    if unique_count <= 5:
        print(
            f"{column}: "
            f"{unique_count} unique values"
        )

# =========================================================
# 4. HIGHLY CORRELATED FEATURES
# =========================================================

print("\n" + "=" * 70)
print("HIGHLY CORRELATED FEATURES")
print("=" * 70)

correlation = X.corr()

found = False

for i in range(len(correlation.columns)):

    for j in range(i + 1, len(correlation.columns)):

        value = correlation.iloc[i, j]

        if abs(value) >= 0.95:

            print(
                f"{correlation.columns[i]} <-> "
                f"{correlation.columns[j]} : "
                f"{value:.4f}"
            )

            found = True

if not found:
    print("No feature pairs with correlation >= 0.95 found.")

print("\n" + "=" * 70)
print("ADVANCED FEATURE ANALYSIS COMPLETE")
print("=" * 70)