
import pandas as pd

train = pd.read_csv(
    "dataset/processed/prepared_v3/X_train.csv"
)

live = pd.read_csv(
    "reports/live_capture_02_v3.csv"
)

raw = pd.read_csv(
    "tools/CICFlowMeter_corrected/data/daily/2026-11-01_Flow.csv"
)

features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Packet Length Mean",
    "Average Packet Size",
    "Flow Bytes/s",
    "Flow Packets/s",
    "SYN Flag Count",
    "ACK Flag Count",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
]

print("LIVE TCP DNS MEDIANS")
live_dns = live[
    (live["Destination Port"] == 53) &
    (raw["Protocol"].reset_index(drop=True) == 6)
]

print(live_dns[features].median())

print("\nLIVE UDP DNS MEDIANS")
live_udp = live[
    (live["Destination Port"] == 53) &
    (raw["Protocol"].reset_index(drop=True) == 17)
]

print(live_udp[features].median())

print("\nTRAINING MEDIANS")
print(train[features].median())
