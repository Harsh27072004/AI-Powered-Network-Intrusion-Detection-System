
import pandas as pd

file = "tools/CICFlowMeter_corrected/data/daily/2026-11-01_Flow.csv"

df = pd.read_csv(file, nrows=5)

print("TCP-RELATED RAW COLUMNS:\n")

for col in df.columns:
    if any(word in col.lower() for word in [
        "win", "window", "cwe", "ece", "flag", "init"
    ]):
        print(repr(col))
