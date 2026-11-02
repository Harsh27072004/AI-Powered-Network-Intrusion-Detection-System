import pandas as pd
import numpy as np

FILE = "dataset/processed/splits/X_train.csv"

print("=" * 70)
print("FEATURE INSPECTION")
print("=" * 70)

print("\nLoading training features...")
X = pd.read_csv(FILE)

print("Loaded successfully.")

print("\nShape:")
print(X.shape)

print("\n" + "=" * 70)
print("FEATURE NAMES")
print("=" * 70)

for i, column in enumerate(X.columns, start=1):
    print(f"{i:2}. {column}")

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(X.dtypes.value_counts())

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = X.isnull().sum()
print(missing[missing > 0])

print("\n" + "=" * 70)
print("INFINITE VALUES")
print("=" * 70)

numeric_X = X.select_dtypes(include=np.number)

infinite = np.isinf(numeric_X).sum()
print(infinite[infinite > 0])

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

constant_features = []

for column in X.columns:

    if X[column].nunique() <= 1:
        constant_features.append(column)

if constant_features:
    for column in constant_features:
        print(column)
else:
    print("No constant features found.")

print("\n" + "=" * 70)
print("FEATURE RANGE")
print("=" * 70)

summary = X.describe().T

print(
    summary[
        ["min", "max", "mean", "std"]
    ].to_string()
)

print("\n" + "=" * 70)
print("FEATURE INSPECTION COMPLETE")
print("=" * 70)