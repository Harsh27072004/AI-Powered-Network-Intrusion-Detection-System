
import pandas as pd

train = pd.read_csv(
    "dataset/processed/prepared_v3/X_train.csv"
)

live = pd.read_csv(
    "reports/live_capture_02_v3.csv"
)

results = []

for col in train.columns:

    train_median = train[col].median()
    live_median = live[col].median()

    difference = abs(live_median - train_median)

    if train_median != 0:
        percentage = difference / abs(train_median) * 100
    else:
        percentage = None

    results.append({
        "Feature": col,
        "Training Median": train_median,
        "Live Median": live_median,
        "Difference": difference,
        "Difference %": percentage
    })

result = pd.DataFrame(results)

result = result.sort_values(
    "Difference %",
    ascending=False,
    na_position="last"
)

print("\nTOP 20 FEATURE DISTRIBUTION DIFFERENCES:\n")
print(result.head(20).to_string(index=False))
