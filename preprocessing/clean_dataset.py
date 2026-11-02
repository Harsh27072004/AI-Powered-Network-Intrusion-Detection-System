import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_DIR = "dataset/MachineLearningCVE"
OUTPUT_DIR = "dataset/processed"

CHUNK_SIZE = 50000

FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Process each CSV file
# ---------------------------------------------------------

for file_name in FILES:

    input_path = os.path.join(INPUT_DIR, file_name)

    output_name = file_name.replace(".csv", "_cleaned.csv")
    output_path = os.path.join(OUTPUT_DIR, output_name)

    print("\n" + "=" * 70)
    print("Processing:", file_name)
    print("=" * 70)

    first_chunk = True
    total_rows = 0
    total_removed = 0

    # Read the file in chunks
    for chunk_number, chunk in enumerate(
        pd.read_csv(input_path, chunksize=CHUNK_SIZE)
    ):

        original_rows = len(chunk)

        # -------------------------------------------------
        # 1. Clean column names
        # -------------------------------------------------

        chunk.columns = chunk.columns.str.strip()

        # -------------------------------------------------
        # 2. Clean labels
        # -------------------------------------------------

        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        # -------------------------------------------------
        # 3. Replace infinity with NaN
        # -------------------------------------------------

        numeric_columns = chunk.select_dtypes(
            include=[np.number]
        ).columns

        chunk[numeric_columns] = chunk[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan
        )

        # -------------------------------------------------
        # 4. Remove rows containing invalid numeric values
        # -------------------------------------------------

        chunk = chunk.dropna(subset=numeric_columns)

        # -------------------------------------------------
        # 5. Remove duplicate rows inside this chunk
        # -------------------------------------------------

        chunk = chunk.drop_duplicates()

        # -------------------------------------------------
        # 6. Create binary target
        # -------------------------------------------------

        chunk["Binary_Label"] = (
            chunk["Label"]
            .apply(lambda x: 0 if x == "BENIGN" else 1)
        )

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        cleaned_rows = len(chunk)

        total_rows += cleaned_rows
        total_removed += original_rows - cleaned_rows

        # -------------------------------------------------
        # Save chunk
        # -------------------------------------------------

        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        print(
            f"Chunk {chunk_number + 1}: "
            f"{original_rows} → {cleaned_rows} rows"
        )

    print("\nFinished:", file_name)
    print("Cleaned rows:", total_rows)
    print("Rows removed:", total_removed)
    print("Saved to:", output_path)


print("\n" + "=" * 70)
print("ALL DATASET FILES PROCESSED")
print("=" * 70)