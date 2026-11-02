import pandas as pd

file_path = "dataset/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

print("Loading dataset...")
print("Please wait...")

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nTraffic labels:")
print(df[" Label"].value_counts())