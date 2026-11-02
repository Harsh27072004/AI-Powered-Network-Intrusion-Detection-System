import pandas as pd
import os

DATASET_DIR = "dataset/processed"

files = [
    "Monday-WorkingHours.pcap_ISCX_cleaned.csv",
    "Tuesday-WorkingHours.pcap_ISCX_cleaned.csv",
    "Wednesday-workingHours.pcap_ISCX_cleaned.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_cleaned.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_cleaned.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX_cleaned.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_cleaned.csv",
   "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_cleaned.csv",
]

print("=" * 70)
print("CLEANED DATASET LABEL CHECK")
print("=" * 70)

for file in files:

    path = os.path.join(DATASET_DIR, file)

    print("\n" + "-" * 70)
    print(file)
    print("-" * 70)

    df = pd.read_csv(
        path,
        usecols=["Label", "Binary_Label"]
    )

    print("\nTotal cleaned records:", len(df))

    print("\nLabel distribution:")
    print(df["Label"].value_counts())

    print("\nBinary distribution:")
    print(df["Binary_Label"].value_counts())

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)