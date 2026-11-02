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
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_cleaned.csv"
]

total_benign = 0
total_attack = 0

print("=" * 70)
print("BINARY DATASET SUMMARY")
print("=" * 70)

for file in files:

    path = os.path.join(DATASET_DIR, file)

    df = pd.read_csv(
        path,
        usecols=["Binary_Label"]
    )

    benign = (df["Binary_Label"] == 0).sum()
    attack = (df["Binary_Label"] == 1).sum()

    total_benign += benign
    total_attack += attack

    print("\n" + file)
    print("BENIGN :", benign)
    print("ATTACK :", attack)

print("\n" + "=" * 70)
print("TOTAL")
print("=" * 70)

print("Total BENIGN :", total_benign)
print("Total ATTACK :", total_attack)
print("Total records:", total_benign + total_attack)

print("=" * 70)