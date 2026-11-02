
import pandas as pd

file = "dataset/processed/binary_nids_dataset_v3.csv"

df = pd.read_csv(file, nrows=5)

print("TOTAL COLUMNS:", len(df.columns))
print("\nCOLUMN NAMES:")

for col in df.columns:
    print(repr(col))
