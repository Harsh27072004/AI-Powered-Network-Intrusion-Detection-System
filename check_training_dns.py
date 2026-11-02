
import pandas as pd
import glob
import os

folder = "dataset/processed"

files = glob.glob(folder + "/*.csv")

total = 0

for file in files:

    try:
        df = pd.read_csv(file)

        if "Destination Port" not in df.columns:
            continue

        if "Label" not in df.columns:
            continue

        dns = df[df["Destination Port"] == 53]

        if dns.empty:
            continue

        print("\nFILE:", os.path.basename(file))
        print("DNS flows:", len(dns))

        print("\nLabels:")
        print(dns["Label"].value_counts().head(10))

        total += len(dns)

    except Exception as e:
        print("Skipped:", file, e)

print("\nTotal DNS flows inspected:", total)
