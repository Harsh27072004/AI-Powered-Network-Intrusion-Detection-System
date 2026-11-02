
import pandas as pd
import numpy as np

file = "reports/live_capture_02_v3.csv"

df = pd.read_csv(file)

print("TOTAL FLOWS:", len(df))
print("TOTAL FEATURES:", len(df.columns))

print("\nMISSING VALUES BY FEATURE:")

missing = df.isna().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No missing values.")
else:
    print(missing.sort_values(ascending=False))

print("\nINFINITE VALUES:")

numeric = df.select_dtypes(include=[np.number])

infinite = np.isinf(numeric).sum()
infinite = infinite[infinite > 0]

if infinite.empty:
    print("No infinite values.")
else:
    print(infinite)

print("\nZERO-VALUE FEATURES:")

zero_counts = (numeric == 0).sum()
print(zero_counts.sort_values(ascending=False).head(15))
