
import pandas as pd

raw_file = "tools/CICFlowMeter_corrected/data/daily/2026-11-01_Flow.csv"
live_file = "reports/live_capture_02_v3.csv"

raw = pd.read_csv(raw_file)
live = pd.read_csv(live_file)

columns = [
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min"
]

print("RAW CICFLOWMETER VS CONVERTED VALUES\n")

for col in columns:
    if col in raw.columns and col in live.columns:
        print("\nFEATURE:", col)
        print("RAW MEDIAN:", raw[col].median())
        print("LIVE MEDIAN:", live[col].median())
        print("RAW SAMPLE:", raw[col].head(3).tolist())
        print("LIVE SAMPLE:", live[col].head(3).tolist())
    else:
        print("\nMISSING:", col)
