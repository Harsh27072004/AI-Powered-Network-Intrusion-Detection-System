
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEDIANS_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "prepared_v3",
    "training_medians.csv"
)

FEATURE_REFERENCE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "prepared_v3",
    "X_train.csv"
)

COLUMN_MAPPING = {
    "Destination Port": "Dst Port",
    "Total Fwd Packets": "Total Fwd Packet",
    "Total Backward Packets": "Total Bwd packets",
    "Total Length of Fwd Packets": "Total Length of Fwd Packet",
    "Total Length of Bwd Packets": "Total Length of Bwd Packet",
    "Min Packet Length": "Packet Length Min",
    "Max Packet Length": "Packet Length Max",
    "Avg Bwd Segment Size": "Bwd Segment Size Avg",
    "Init_Win_bytes_forward": "FWD Init Win Bytes",
    "Init_Win_bytes_backward": "Bwd Init Win Bytes",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
    "CWE Flag Count": "CWR Flag Count",
}


def convert(input_file, output_file):

    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"Input CSV not found: {input_file}"
        )

    raw = pd.read_csv(input_file)

    # Load expected model feature order
    feature_order = pd.read_csv(
        FEATURE_REFERENCE,
        nrows=0
    ).columns.tolist()

    # Load training medians
    median_data = pd.read_csv(MEDIANS_FILE)

    medians = median_data.set_index(
        "Unnamed: 0"
    )["median"]

    if len(feature_order) != 61:
        raise ValueError(
            f"Expected 61 features, found {len(feature_order)}"
        )

    if (
        len(medians) != 61
        or set(medians.index) != set(feature_order)
    ):
        raise ValueError(
            "Training medians do not match the V3 feature list"
        )

    # Rename CICFlowMeter columns
    raw = raw.rename(
        columns={
            source: target
            for target, source in COLUMN_MAPPING.items()
        }
    )

    # Validate required model features
    missing = [
        feature
        for feature in feature_order
        if feature not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required CICFlowMeter columns: {missing}"
        )

    # Select exactly the 61 model features
    data = raw[feature_order].copy()

    # Preserve protocol as additional metadata
    if "Protocol" not in raw.columns:
        raise ValueError(
            "Protocol column missing from CICFlowMeter input."
        )

    protocol = pd.to_numeric(
        raw["Protocol"],
        errors="coerce"
    )

    if protocol.isna().any():
        raise ValueError(
            "Invalid or missing protocol values detected."
        )

    # Convert model features to numeric
    for column in feature_order:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Replace infinite values
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    missing_before = int(
        data.isna().sum().sum()
    )

    # Median imputation for model features only
    data = data.fillna(
        medians[feature_order]
    )

    if not np.isfinite(
        data.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values remain after median imputation"
        )

    # Add protocol metadata AFTER feature preprocessing
    data["Protocol"] = protocol.to_numpy()

    # Save output
    output_directory = os.path.dirname(output_file)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    data.to_csv(
        output_file,
        index=False
    )

    print("\nCICFlowMeter to V3 conversion successful")
    print("Input flows:", len(raw))
    print("Output flows:", len(data))
    print("Model features:", len(feature_order))
    print("Total output columns:", len(data.columns))
    print("Missing values replaced:", missing_before)

    print("\nProtocol distribution:")
    print(data["Protocol"].value_counts().to_string())

    print("\nOutput:", output_file)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python preprocessing/cicflowmeter_to_v3.py "
            "<input_csv> <output_csv>"
        )
        sys.exit(1)

    convert(
        sys.argv[1],
        sys.argv[2]
    )
