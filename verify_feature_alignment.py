
import pandas as pd

train_file = "dataset/processed/prepared_v3/X_train.csv"
live_file = "reports/live_capture_02_v3.csv"

train = pd.read_csv(train_file, nrows=1)
live = pd.read_csv(live_file, nrows=1)

train_features = list(train.columns)
live_features = list(live.columns)

print("TRAINING FEATURE COUNT:", len(train_features))
print("LIVE FEATURE COUNT:", len(live_features))

print("\nFEATURE ORDER MATCH:", train_features == live_features)

missing = [f for f in train_features if f not in live_features]
extra = [f for f in live_features if f not in train_features]

print("\nMISSING LIVE FEATURES:", missing)
print("EXTRA LIVE FEATURES:", extra)

if train_features != live_features:
    print("\nPOSITIONAL DIFFERENCES:")

    for i, (a, b) in enumerate(
        zip(train_features, live_features)
    ):
        if a != b:
            print(f"Position {i}: TRAIN={a} | LIVE={b}")

if not missing and not extra and train_features == live_features:
    print("\nRESULT: FEATURE NAMES AND ORDER MATCH.")
else:
    print("\nRESULT: FEATURE ALIGNMENT NEEDS CORRECTION.")
