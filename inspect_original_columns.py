
import pandas as pd

file = "dataset/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(file, nrows=5, encoding="utf-8-sig")

print("ORIGINAL COLUMN HEADERS:\n")

for col in df.columns:
    print(repr(col))

print("\nTOTAL COLUMNS:", len(df.columns))
