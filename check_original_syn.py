import pandas as pd
import glob
import os

files = glob.glob("dataset/cleaned/*.csv")

if not files:
    files = glob.glob("dataset/**/*.csv", recursive=True)

for file in files:
    try:
        df = pd.read_csv(file, usecols=["SYN Flag Count"])

        print("\nFile:", os.path.basename(file))
        print(df["SYN Flag Count"].value_counts(dropna=False).sort_index())

        unusual = df[df["SYN Flag Count"] > 1]

        print("Values greater than 1:", len(unusual))

    except Exception:
        continue