import pandas as pd
import numpy as np

file_path = "dataset/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"

print("Loading dataset...")
df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

# Clean column names temporarily for inspection
df.columns = df.columns.str.strip()

print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n========== INFINITE VALUES ==========")
numeric_df = df.select_dtypes(include=np.number)

infinite_values = np.isinf(numeric_df).sum()
print(infinite_values[infinite_values > 0])

print("\n========== DUPLICATE ROWS ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== DATA TYPES ==========")
print(df.dtypes.value_counts())

print("\n========== LABEL DISTRIBUTION ==========")
print(df["Label"].value_counts())

print("\n========== DATA INSPECTION COMPLETE ==========")