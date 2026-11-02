
import pandas as pd

raw_file = "tools/CICFlowMeter_corrected/data/daily/2026-11-01_Flow.csv"
live_file = "reports/live_capture_02_v3.csv"

raw = pd.read_csv(raw_file)
live = pd.read_csv(live_file)

columns = [
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
]

for col in columns:
    if col in raw.columns and col in live.columns:
        print("\nFEATURE:", col)
        print("RAW MEDIAN:", raw[col].median())
        print("LIVE MEDIAN:", live[col].median())
        print("RAW SAMPLE:", raw[col].head(5).tolist())
        print("LIVE SAMPLE:", live[col].head(5).tolist())
    else:
        print("\nMISSING:", col)
