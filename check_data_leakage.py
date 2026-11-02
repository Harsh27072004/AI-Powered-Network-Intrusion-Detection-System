
import pandas as pd

path = "dataset/processed/binary_nids_dataset_v3.csv"

print("Loading dataset...")
df = pd.read_csv(path, low_memory=False)

# Separate labels from features
y = df["Binary_Label"]
X = df.drop(columns=["Label", "Binary_Label"])

# Count identical feature patterns
print("\nTotal rows:", len(df))

duplicate_mask = X.duplicated(keep=False)
print("Rows belonging to duplicate feature patterns:", duplicate_mask.sum())

# Check whether identical features have conflicting labels
temp = X.copy()
temp["Binary_Label"] = y.values

label_counts = temp.groupby(
    list(X.columns),
    dropna=False
)["Binary_Label"].nunique()

conflicting_groups = (label_counts > 1).sum()

print("Unique feature patterns:", len(label_counts))
print("Patterns with conflicting labels:", conflicting_groups)

# Examine important features by class
important_features = [
    "Packet Length Variance",
    "Average Packet Size",
    "Packet Length Std",
    "Packet Length Mean",
    "Destination Port"
]

print("\nFEATURE STATISTICS BY CLASS")

for feature in important_features:
    if feature in X.columns:
        print("\n", feature)
        print(
            df.groupby("Binary_Label")[feature]
            .agg(["min", "median", "mean", "max"])
            .to_string()
        )

print("\nCHECK COMPLETED")
