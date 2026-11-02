
import pandas as pd

live = pd.read_csv("reports/live_capture_02_v3.csv")
train = pd.read_csv(
    "dataset/processed/prepared_v3/X_train.csv",
    nrows=50000
)

print("LIVE FLOWS:", len(live))
print("TRAINING FLOWS:", len(train))

print("\nFEATURE COMPARISON")
print("-" * 90)

for col in live.columns:

    if col not in train.columns:
        continue

    a = pd.to_numeric(live[col], errors="coerce").dropna()
    b = pd.to_numeric(train[col], errors="coerce").dropna()

    if a.empty or b.empty:
        continue

    print(f"\n{col}")
    print(f"Live   -> Median: {a.median():.4f}, Min: {a.min():.4f}, Max: {a.max():.4f}")
    print(f"Training -> Median: {b.median():.4f}, Min: {b.min():.4f}, Max: {b.max():.4f}")
