
import pandas as pd
import joblib

train = pd.read_csv(
    r"dataset\processed\binary_nids_dataset_v3.csv",
    nrows=50000
)

test = pd.read_csv(
    r"dataset\processed\cicflowmeter_test\test_capture_v3.csv"
)

model = joblib.load(
    r"models\random_forest_binary_v3.pkl"
)

features = list(model.feature_names_in_)

rows = []

for c in features:
    a = pd.to_numeric(train[c], errors="coerce").dropna()
    b = pd.to_numeric(test[c], errors="coerce").dropna()

    lo = a.min()
    hi = a.max()

    outside = ((b < lo) | (b > hi)).sum()

    rows.append({
        "Feature": c,
        "Train_min": round(lo, 3),
        "Train_max": round(hi, 3),
        "Test_min": round(b.min(), 3),
        "Test_max": round(b.max(), 3),
        "Outside_train_range": int(outside)
    })

result = pd.DataFrame(rows)

print("Training sample:", len(train))
print("Test flows:", len(test))

print(
    "\nFeatures outside training range:",
    (result["Outside_train_range"] > 0).sum(),
    "of", len(features)
)

print("\nLargest range mismatches:")

print(
    result.sort_values(
        "Outside_train_range",
        ascending=False
    ).head(15).to_string(index=False)
)
