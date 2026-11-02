import pandas as pd
import os

DATASET_DIR = "dataset/MachineLearningCVE"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]

print("=" * 70)
print("CIC-IDS2017 DATASET SUMMARY")
print("=" * 70)

for file in files:

    path = os.path.join(DATASET_DIR, file)

    print("\n" + "-" * 70)
    print(file)
    print("-" * 70)

    # Read only the Label column.
    # This saves memory because we don't need all 79 columns yet.
    df = pd.read_csv(path, usecols=[" Label"])

    df.columns = df.columns.str.strip()

    print("Total records:", len(df))
    print("\nTraffic labels:")

    print(df["Label"].value_counts())

print("\n" + "=" * 70)
print("SUMMARY COMPLETE")
print("=" * 70)