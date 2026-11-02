
import pandas as pd

path = r"dataset\processed\binary_nids_dataset_v3.csv"

df = pd.read_csv(
    path,
    usecols=["SYN Flag Count", "Label"]
)

print("SYN Flag Count distribution by label:\n")

print(
    pd.crosstab(
        df["SYN Flag Count"],
        df["Label"],
        margins=True
    ).to_string()
)

print("\nSYN Flag Count statistics by label:\n")

print(
    df.groupby("Label")["SYN Flag Count"]
    .describe()
    .to_string()
)
