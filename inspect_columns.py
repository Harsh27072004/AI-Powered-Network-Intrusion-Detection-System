
import pandas as pd

file = "dataset/processed/Monday-WorkingHours.pcap_ISCX_cleaned.csv"

df = pd.read_csv(file, nrows=5)

print("ACTUAL COLUMN NAMES:\n")

for col in df.columns:
    print(repr(col))

print("\nTOTAL COLUMNS:", len(df.columns))
