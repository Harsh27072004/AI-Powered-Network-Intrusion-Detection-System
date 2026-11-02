
import pandas as pd

path = r"dataset\processed\binary_nids_dataset_v3.csv"

counts = {}
total = 0

for chunk in pd.read_csv(
    path,
    usecols=["SYN Flag Count"],
    chunksize=500000
):
    values = chunk["SYN Flag Count"].value_counts()

    for value, count in values.items():
        counts[value] = counts.get(value, 0) + int(count)

    total += len(chunk)

print("Total training rows:", total)
print("\nFull training SYN Flag Count distribution:")

print(
    pd.Series(counts)
    .sort_index()
    .to_string()
)
