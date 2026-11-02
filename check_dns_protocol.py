
import pandas as pd
import glob
import os

files = glob.glob("dataset/MachineLearningCVE/*.csv")
all_dns = []

for file in files:
    print("Reading:", os.path.basename(file))

    df = pd.read_csv(file, low_memory=False, encoding="utf-8-sig")

    # Remove leading and trailing spaces from every column
    df.columns = df.columns.str.strip()

    required = ["Destination Port", "Protocol", "Label"]

    if not all(c in df.columns for c in required):
        print("Missing columns:", required)
        continue

    # Select DNS flows
    dns = df[df["Destination Port"] == 53]

    if not dns.empty:
        all_dns.append(dns[["Protocol", "Label"]])

if not all_dns:
    print("No DNS records found.")
else:
    data = pd.concat(all_dns, ignore_index=True)

    data["Label"] = data["Label"].astype(str).str.strip()

    print("\nDNS PROTOCOL VS LABEL:")
    print(pd.crosstab(data["Protocol"], data["Label"]))

    print("\nBENIGN DNS BY PROTOCOL:")
    print(data[data["Label"] == "BENIGN"]["Protocol"].value_counts())

    print("\nTCP DNS LABELS:")
    print(data[data["Protocol"] == 6]["Label"].value_counts())

    print("\nUDP DNS LABELS:")
    print(data[data["Protocol"] == 17]["Label"].value_counts())
