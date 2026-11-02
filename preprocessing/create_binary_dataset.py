import pandas as pd
import os

# =========================================================
# SETTINGS
# =========================================================

INPUT_DIR = "dataset/processed"
OUTPUT_FILE = "dataset/processed/binary_nids_dataset.csv"

# We have 455,092 attack records.
# We will use all available attack records and
# approximately the same number of benign records.
TARGET_BENIGN = 455092

CHUNK_SIZE = 50000

FILES = [
    "Monday-WorkingHours.pcap_ISCX_cleaned.csv",
    "Tuesday-WorkingHours.pcap_ISCX_cleaned.csv",
    "Wednesday-workingHours.pcap_ISCX_cleaned.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_cleaned.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_cleaned.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX_cleaned.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_cleaned.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_cleaned.csv"
]

# =========================================================
# PREPARE OUTPUT
# =========================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

benign_parts = []
attack_parts = []

benign_count = 0
attack_count = 0

print("=" * 70)
print("CREATING BINARY NIDS DATASET")
print("=" * 70)

# =========================================================
# READ CLEANED FILES
# =========================================================

for file_name in FILES:

    path = os.path.join(INPUT_DIR, file_name)

    print("\nProcessing:", file_name)

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):

        # -------------------------------------------------
        # ATTACK RECORDS
        # -------------------------------------------------

        attacks = chunk[chunk["Binary_Label"] == 1]

        if len(attacks) > 0:
            attack_parts.append(attacks)
            attack_count += len(attacks)

        # -------------------------------------------------
        # BENIGN RECORDS
        # -------------------------------------------------

        remaining_benign = TARGET_BENIGN - benign_count

        if remaining_benign > 0:

            benign = chunk[chunk["Binary_Label"] == 0]

            if len(benign) > remaining_benign:
                benign = benign.iloc[:remaining_benign]

            if len(benign) > 0:
                benign_parts.append(benign)
                benign_count += len(benign)

        print(
            "Current BENIGN:",
            benign_count,
            "| Current ATTACK:",
            attack_count
        )

# =========================================================
# COMBINE
# =========================================================

print("\nCombining selected records...")

benign_df = pd.concat(benign_parts, ignore_index=True)
attack_df = pd.concat(attack_parts, ignore_index=True)

# =========================================================
# SHUFFLE
# =========================================================

print("Shuffling dataset...")

final_df = pd.concat(
    [benign_df, attack_df],
    ignore_index=True
)

final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# =========================================================
# SAVE
# =========================================================

print("\nSaving dataset...")

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================================================
# FINAL INFORMATION
# =========================================================

print("\n" + "=" * 70)
print("BINARY DATASET CREATED")
print("=" * 70)

print("Total records:", len(final_df))
print("BENIGN:", (final_df["Binary_Label"] == 0).sum())
print("ATTACK:", (final_df["Binary_Label"] == 1).sum())

print("\nDataset saved at:")
print(OUTPUT_FILE)

print("=" * 70)