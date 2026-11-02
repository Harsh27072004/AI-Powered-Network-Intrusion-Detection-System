import pandas as pd

file = "tools/CICFlowMeter_corrected/data/daily/2026-11-01_Flow.csv"
df = pd.read_csv(file)

print("Total flows:", len(df))
print("\nProtocol distribution:")
print(df["Protocol"].value_counts())

print("\nFlag count statistics:")

for col in ["SYN Flag Count", "ACK Flag Count", "FIN Flag Count", "RST Flag Count"]:
    if col in df.columns:
        print(f"\n{col}")
        print(df[col].describe())
        print("Non-zero flows:", (df[col] > 0).sum())
    else:
        print(col, "NOT FOUND")