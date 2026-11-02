import subprocess
import pandas as pd
import os
import math
from collections import defaultdict


# ============================================================
# AI-NIDS FLOW EXTRACTOR
# CICIDS2017 / CICFlowMeter-style 61 FEATURES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PCAP_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "test_capture.pcap"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "live_flows.csv"
)

TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"

# Flow timeout in seconds
FLOW_TIMEOUT = 120.0


# ============================================================
# EXACT 61 FEATURES EXPECTED BY THE MODEL
# ============================================================

EXPECTED_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
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
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Bwd Segment Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min"
]


# ============================================================
# TSHARK FIELDS
#
# IMPORTANT:
# ipv6.hdr_len is NOT used because the installed TShark
# version reports it as an invalid field.
# ============================================================

FIELDS = [
    "frame.time_epoch",
    "frame.len",

    "ip.hdr_len",

    "tcp.window_size_value",
    "tcp.hdr_len",
    "tcp.len",

    "ip.src",
    "ip.dst",
    "ipv6.src",
    "ipv6.dst",

    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",

    "tcp.flags.syn",
    "tcp.flags.fin",
    "tcp.flags.reset",
    "tcp.flags.push",
    "tcp.flags.ack",
    "tcp.flags.urg"
]


# ============================================================
# SAFE CONVERSION FUNCTIONS
# ============================================================

def to_float(value, default=0.0):

    try:

        if value is None:
            return default

        value = str(value).strip()

        if value == "":
            return default

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (ValueError, TypeError):

        return default


def to_int(value, default=0):

    try:

        if value is None:
            return default

        value = str(value).strip()

        if value == "":
            return default

        return int(float(value))

    except (ValueError, TypeError):

        return default


def flag_to_int(value):

    if value is None:
        return 0

    value = str(value).strip().lower()

    if value in (
        "1",
        "true",
        "yes"
    ):
        return 1

    return 0


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def safe_mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def safe_min(values):

    if not values:
        return 0.0

    return min(values)


def safe_max(values):

    if not values:
        return 0.0

    return max(values)


def safe_std(values):

    if len(values) < 2:
        return 0.0

    avg = safe_mean(values)

    result = sum(
        (x - avg) ** 2
        for x in values
    ) / len(values)

    return math.sqrt(
        max(result, 0.0)
    )


def safe_variance(values):

    if not values:
        return 0.0

    avg = safe_mean(values)

    return sum(
        (x - avg) ** 2
        for x in values
    ) / len(values)


def safe_stats(values):

    if not values:

        return (
            0.0,
            0.0,
            0.0,
            0.0
        )

    return (
        safe_mean(values),
        safe_std(values),
        safe_max(values),
        safe_min(values)
    )


# ============================================================
# READ PCAP USING TSHARK
# ============================================================

def read_pcap():

    print("=" * 70)
    print("AI-NIDS FLOW EXTRACTOR")
    print("=" * 70)

    print("\nReading PCAP:")
    print(PCAP_FILE)

    # --------------------------------------------------------
    # Check PCAP
    # --------------------------------------------------------

    if not os.path.exists(PCAP_FILE):

        raise FileNotFoundError(
            "PCAP file not found:\n"
            + PCAP_FILE
        )

    # --------------------------------------------------------
    # Check TShark
    # --------------------------------------------------------

    if not os.path.exists(TSHARK_PATH):

        raise FileNotFoundError(
            "TShark executable not found:\n"
            + TSHARK_PATH
        )

    # --------------------------------------------------------
    # Build command
    # --------------------------------------------------------

    command = [
        TSHARK_PATH,
        "-r",
        PCAP_FILE,
        "-T",
        "fields"
    ]

    for field in FIELDS:

        command.extend([
            "-e",
            field
        ])

    command.extend([
        "-E",
        "separator=\t",

        "-E",
        "quote=n",

        "-E",
        "header=n",

        "-E",
        "occurrence=f"
    ])

    # --------------------------------------------------------
    # Execute TShark
    # --------------------------------------------------------

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # --------------------------------------------------------
    # Handle error
    # --------------------------------------------------------

    if result.returncode != 0:

        print("\nTShark error:")
        print(result.stderr)

        raise RuntimeError(
            "TShark failed to read the PCAP."
        )

    # --------------------------------------------------------
    # Extract lines
    # --------------------------------------------------------

    lines = result.stdout.splitlines()

    print(
        "Packets read:",
        len(lines)
    )

    return lines


# ============================================================
# PARSE PACKETS
# ============================================================

def parse_packets(lines):

    packets = []

    for line in lines:

        values = line.split("\t")

        if len(values) != len(FIELDS):

            continue

        (
            timestamp,
            frame_length,

            ip_header_length,

            tcp_window,
            tcp_header_length,
            tcp_payload_length,

            ip_src,
            ip_dst,
            ipv6_src,
            ipv6_dst,

            tcp_src,
            tcp_dst,
            udp_src,
            udp_dst,

            syn,
            fin,
            rst,
            psh,
            ack,
            urg

        ) = values

        # ----------------------------------------------------
        # IP addresses
        # ----------------------------------------------------

        src_ip = (
            ip_src.strip()
            if ip_src.strip()
            else ipv6_src.strip()
        )

        dst_ip = (
            ip_dst.strip()
            if ip_dst.strip()
            else ipv6_dst.strip()
        )

        if not src_ip or not dst_ip:

            continue

        # ----------------------------------------------------
        # Protocol
        # ----------------------------------------------------

        if tcp_src and tcp_dst:

            protocol = "TCP"

            src_port = to_int(
                tcp_src
            )

            dst_port = to_int(
                tcp_dst
            )

        elif udp_src and udp_dst:

            protocol = "UDP"

            src_port = to_int(
                udp_src
            )

            dst_port = to_int(
                udp_dst
            )

        else:

            # Ignore ICMP / ICMPv6 / ARP.
            continue

        # ----------------------------------------------------
        # Numeric fields
        # ----------------------------------------------------

        timestamp = to_float(
            timestamp
        )

        frame_length = to_float(
            frame_length
        )

        ip_header_length = to_float(
            ip_header_length
        )

        tcp_window = to_float(
            tcp_window
        )

        tcp_header_length = to_float(
            tcp_header_length
        )

        tcp_payload_length = to_float(
            tcp_payload_length
        )

        # ----------------------------------------------------
        # Payload length
        #
        # tcp.len is TCP payload length.
        # For UDP, estimate payload from frame length.
        # ----------------------------------------------------

        if protocol == "TCP":

            payload_length = max(
                tcp_payload_length,
                0.0
            )

        else:

            # Ethernet 14 + IP 20 + UDP 8
            estimated_payload = (
                frame_length
                - 14.0
                - 20.0
                - 8.0
            )

            payload_length = max(
                estimated_payload,
                0.0
            )

        # ----------------------------------------------------
        # TCP segment size
        # ----------------------------------------------------

        if protocol == "TCP":

            tcp_segment_size = max(
                tcp_header_length
                + payload_length,
                0.0
            )

        else:

            tcp_segment_size = 0.0

        # ----------------------------------------------------
        # Packet
        # ----------------------------------------------------

        packet = {

            "timestamp":
                timestamp,

            "frame_length":
                frame_length,

            "payload_length":
                payload_length,

            "tcp_segment_size":
                tcp_segment_size,

            "ip_header_length":
                ip_header_length,

            "tcp_window":
                tcp_window,

            "tcp_header_length":
                tcp_header_length,

            "src_ip":
                src_ip,

            "dst_ip":
                dst_ip,

            "src_port":
                src_port,

            "dst_port":
                dst_port,

            "protocol":
                protocol,

            "syn":
                flag_to_int(syn),

            "fin":
                flag_to_int(fin),

            "rst":
                flag_to_int(rst),

            "psh":
                flag_to_int(psh),

            "ack":
                flag_to_int(ack),

            "urg":
                flag_to_int(urg)
        }

        packets.append(
            packet
        )

    print(
        "TCP/UDP packets:",
        len(packets)
    )

    return packets


# ============================================================
# FLOW KEY
# ============================================================

def flow_key(packet):

    endpoint_a = (
        packet["src_ip"],
        packet["src_port"]
    )

    endpoint_b = (
        packet["dst_ip"],
        packet["dst_port"]
    )

    ordered = sorted([
        endpoint_a,
        endpoint_b
    ])

    return (
        ordered[0][0],
        ordered[0][1],
        ordered[1][0],
        ordered[1][1],
        packet["protocol"]
    )


# ============================================================
# BUILD FLOWS
# ============================================================

def build_flows(packets):

    packets = sorted(
        packets,
        key=lambda x: x["timestamp"]
    )

    flows = defaultdict(list)

    last_seen = {}

    flow_number = defaultdict(int)

    for packet in packets:

        base_key = flow_key(
            packet
        )

        timestamp = packet[
            "timestamp"
        ]

        # ----------------------------------------------------
        # Split same 5-tuple when timeout is exceeded.
        # ----------------------------------------------------

        if base_key in last_seen:

            gap = (
                timestamp
                - last_seen[base_key]
            )

            if gap > FLOW_TIMEOUT:

                flow_number[
                    base_key
                ] += 1

        current_number = flow_number[
            base_key
        ]

        complete_key = (
            base_key,
            current_number
        )

        flows[
            complete_key
        ].append(packet)

        last_seen[
            base_key
        ] = timestamp

    print(
        "Unique flows:",
        len(flows)
    )

    return flows


# ============================================================
# SPLIT FORWARD / BACKWARD
# ============================================================

def split_direction(packets):

    first = packets[0]

    src_ip = first[
        "src_ip"
    ]

    src_port = first[
        "src_port"
    ]

    dst_ip = first[
        "dst_ip"
    ]

    dst_port = first[
        "dst_port"
    ]

    forward = []

    backward = []

    for packet in packets:

        same_direction = (
            packet["src_ip"] == src_ip
            and
            packet["src_port"] == src_port
            and
            packet["dst_ip"] == dst_ip
            and
            packet["dst_port"] == dst_port
        )

        if same_direction:

            forward.append(
                packet
            )

        else:

            backward.append(
                packet
            )

    return (
        forward,
        backward
    )


# ============================================================
# INTER ARRIVAL TIMES
# ============================================================

def calculate_iats(packets):

    if len(packets) < 2:

        return []

    packets = sorted(
        packets,
        key=lambda x: x["timestamp"]
    )

    timestamps = [
        p["timestamp"]
        for p in packets
    ]

    result = []

    for i in range(1, len(timestamps)):

        difference = (
            timestamps[i]
            - timestamps[i - 1]
        )

        if difference >= 0:

            result.append(
                difference
            )

    return result


# ============================================================
# ACTIVE / IDLE
# ============================================================

def calculate_active_idle(
    timestamps
):

    if len(timestamps) < 2:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        )

    timestamps = sorted(
        timestamps
    )

    # CICFlowMeter-style active timeout.
    threshold = 1.0

    active_periods = []

    idle_periods = []

    active_start = timestamps[0]

    previous = timestamps[0]

    for current in timestamps[1:]:

        gap = (
            current
            - previous
        )

        if gap < 0:

            previous = current

            continue

        if gap > threshold:

            active_duration = (
                previous
                - active_start
            )

            if active_duration > 0:

                active_periods.append(
                    active_duration
                )

            idle_periods.append(
                gap
            )

            active_start = current

        previous = current

    # Final active period.

    active_duration = (
        previous
        - active_start
    )

    if active_duration > 0:

        active_periods.append(
            active_duration
        )

    # Convert seconds → microseconds.

    active_us = [
        x * 1_000_000
        for x in active_periods
    ]

    idle_us = [
        x * 1_000_000
        for x in idle_periods
    ]

    (
        active_mean,
        active_std,
        active_max,
        active_min
    ) = safe_stats(
        active_us
    )

    (
        idle_mean,
        idle_std,
        idle_max,
        idle_min
    ) = safe_stats(
        idle_us
    )

    return (
        active_mean,
        active_std,
        active_max,
        active_min,
        idle_mean,
        idle_std,
        idle_max,
        idle_min
    )


# ============================================================
# CALCULATE 61 FEATURES
# ============================================================

def calculate_features(
    flows
):

    results = []

    for flow_id, packets in flows.items():

        if not packets:

            continue

        packets = sorted(
            packets,
            key=lambda x: x["timestamp"]
        )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        (
            forward,
            backward
        ) = split_direction(
            packets
        )

        # ----------------------------------------------------
        # Packet lengths
        # ----------------------------------------------------

        forward_lengths = [
            p["payload_length"]
            for p in forward
        ]

        backward_lengths = [
            p["payload_length"]
            for p in backward
        ]

        all_lengths = (
            forward_lengths
            +
            backward_lengths
        )

        # ----------------------------------------------------
        # Frame lengths
        # ----------------------------------------------------

        frame_lengths = [
            p["frame_length"]
            for p in packets
        ]

        # ----------------------------------------------------
        # Timestamps
        # ----------------------------------------------------

        timestamps = [
            p["timestamp"]
            for p in packets
        ]

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration_seconds = (
            timestamps[-1]
            -
            timestamps[0]
        )

        duration_seconds = max(
            duration_seconds,
            0.0
        )

        duration_us = (
            duration_seconds
            *
            1_000_000
        )

        # ----------------------------------------------------
        # IAT
        # ----------------------------------------------------

        flow_iat = calculate_iats(
            packets
        )

        forward_iat = calculate_iats(
            forward
        )

        backward_iat = calculate_iats(
            backward
        )

        # ----------------------------------------------------
        # Active / Idle
        # ----------------------------------------------------

        (
            active_mean,
            active_std,
            active_max,
            active_min,
            idle_mean,
            idle_std,
            idle_max,
            idle_min

        ) = calculate_active_idle(
            timestamps
        )

        # ----------------------------------------------------
        # Rates
        # ----------------------------------------------------

        if duration_seconds > 0:

            flow_bytes_s = (
                sum(frame_lengths)
                /
                duration_seconds
            )

            flow_packets_s = (
                len(packets)
                /
                duration_seconds
            )

            fwd_packets_s = (
                len(forward)
                /
                duration_seconds
            )

            bwd_packets_s = (
                len(backward)
                /
                duration_seconds
            )

        else:

            flow_bytes_s = 0.0
            flow_packets_s = 0.0
            fwd_packets_s = 0.0
            bwd_packets_s = 0.0

        # ----------------------------------------------------
        # Flags
        # ----------------------------------------------------

        fin_count = sum(
            p["fin"]
            for p in packets
        )

        syn_count = sum(
            p["syn"]
            for p in packets
        )

        rst_count = sum(
            p["rst"]
            for p in packets
        )

        psh_count = sum(
            p["psh"]
            for p in packets
        )

        ack_count = sum(
            p["ack"]
            for p in packets
        )

        urg_count = sum(
            p["urg"]
            for p in packets
        )

        fwd_psh_flags = sum(
            p["psh"]
            for p in forward
        )

        # ----------------------------------------------------
        # Header lengths
        # ----------------------------------------------------

        fwd_header_length = sum(
            p["tcp_header_length"]
            for p in forward
        )

        bwd_header_length = sum(
            p["tcp_header_length"]
            for p in backward
        )

        # ----------------------------------------------------
        # Initial windows
        # ----------------------------------------------------

        if forward:

            init_win_forward = (
                forward[0][
                    "tcp_window"
                ]
            )

        else:

            init_win_forward = 0.0

        if backward:

            init_win_backward = (
                backward[0][
                    "tcp_window"
                ]
            )

        else:

            init_win_backward = 0.0

        # ----------------------------------------------------
        # Active forward data packets
        # ----------------------------------------------------

        act_data_pkt_fwd = sum(
            1
            for p in forward
            if p["payload_length"] > 0
        )

        # ----------------------------------------------------
        # Minimum forward TCP segment size
        # ----------------------------------------------------

        forward_segment_sizes = [
            p["tcp_segment_size"]
            for p in forward
            if p["tcp_segment_size"] > 0
        ]

        if forward_segment_sizes:

            min_seg_size_forward = min(
                forward_segment_sizes
            )

        else:

            min_seg_size_forward = 0.0

        # ----------------------------------------------------
        # Down / Up
        # ----------------------------------------------------

        if len(forward) > 0:

            down_up_ratio = (
                len(backward)
                /
                len(forward)
            )

        else:

            down_up_ratio = 0.0

        # ----------------------------------------------------
        # Average packet sizes
        # ----------------------------------------------------

        average_packet_size = safe_mean(
            all_lengths
        )

        avg_bwd_segment_size = safe_mean(
            backward_lengths
        )

        # ----------------------------------------------------
        # Build exact feature dictionary
        # ----------------------------------------------------

        feature = {

            "Destination Port":
                packets[0]["dst_port"],

            "Flow Duration":
                duration_us,

            "Total Fwd Packets":
                len(forward),

            "Total Backward Packets":
                len(backward),

            "Total Length of Fwd Packets":
                sum(forward_lengths),

            "Total Length of Bwd Packets":
                sum(backward_lengths),

            "Fwd Packet Length Max":
                safe_max(forward_lengths),

            "Fwd Packet Length Min":
                safe_min(forward_lengths),

            "Fwd Packet Length Mean":
                safe_mean(forward_lengths),

            "Fwd Packet Length Std":
                safe_std(forward_lengths),

            "Bwd Packet Length Max":
                safe_max(backward_lengths),

            "Bwd Packet Length Min":
                safe_min(backward_lengths),

            "Bwd Packet Length Mean":
                safe_mean(backward_lengths),

            "Bwd Packet Length Std":
                safe_std(backward_lengths),

            "Flow Bytes/s":
                flow_bytes_s,

            "Flow Packets/s":
                flow_packets_s,

            "Flow IAT Mean":
                safe_mean(flow_iat)
                * 1_000_000,

            "Flow IAT Std":
                safe_std(flow_iat)
                * 1_000_000,

            "Flow IAT Max":
                safe_max(flow_iat)
                * 1_000_000,

            "Flow IAT Min":
                safe_min(flow_iat)
                * 1_000_000,

            "Fwd IAT Total":
                sum(forward_iat)
                * 1_000_000,

            "Fwd IAT Mean":
                safe_mean(forward_iat)
                * 1_000_000,

            "Fwd IAT Std":
                safe_std(forward_iat)
                * 1_000_000,

            "Fwd IAT Max":
                safe_max(forward_iat)
                * 1_000_000,

            "Fwd IAT Min":
                safe_min(forward_iat)
                * 1_000_000,

            "Bwd IAT Total":
                sum(backward_iat)
                * 1_000_000,

            "Bwd IAT Mean":
                safe_mean(backward_iat)
                * 1_000_000,

            "Bwd IAT Std":
                safe_std(backward_iat)
                * 1_000_000,

            "Bwd IAT Max":
                safe_max(backward_iat)
                * 1_000_000,

            "Bwd IAT Min":
                safe_min(backward_iat)
                * 1_000_000,

            "Fwd PSH Flags":
                fwd_psh_flags,

            "Fwd Header Length":
                fwd_header_length,

            "Bwd Header Length":
                bwd_header_length,

            "Fwd Packets/s":
                fwd_packets_s,

            "Bwd Packets/s":
                bwd_packets_s,

            "Min Packet Length":
                safe_min(all_lengths),

            "Max Packet Length":
                safe_max(all_lengths),

            "Packet Length Mean":
                safe_mean(all_lengths),

            "Packet Length Std":
                safe_std(all_lengths),

            "Packet Length Variance":
                safe_variance(all_lengths),

            "FIN Flag Count":
                fin_count,

            "SYN Flag Count":
                syn_count,

            "RST Flag Count":
                rst_count,

            "PSH Flag Count":
                psh_count,

            "ACK Flag Count":
                ack_count,

            "URG Flag Count":
                urg_count,

            "Down/Up Ratio":
                down_up_ratio,

            "Average Packet Size":
                average_packet_size,

            "Avg Bwd Segment Size":
                avg_bwd_segment_size,

            "Init_Win_bytes_forward":
                init_win_forward,

            "Init_Win_bytes_backward":
                init_win_backward,

            "act_data_pkt_fwd":
                act_data_pkt_fwd,

            "min_seg_size_forward":
                min_seg_size_forward,

            "Active Mean":
                active_mean,

            "Active Std":
                active_std,

            "Active Max":
                active_max,

            "Active Min":
                active_min,

            "Idle Mean":
                idle_mean,

            "Idle Std":
                idle_std,

            "Idle Max":
                idle_max,

            "Idle Min":
                idle_min
        }

        results.append(
            feature
        )

    return pd.DataFrame(
        results,
        columns=EXPECTED_FEATURES
    )


# ============================================================
# CLEAN AND VALIDATE OUTPUT
# ============================================================

def validate_output(df):

    print("\n" + "=" * 70)
    print("OUTPUT VALIDATION")
    print("=" * 70)

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    # --------------------------------------------------------
    # Missing columns
    # --------------------------------------------------------

    missing = [
        column
        for column in EXPECTED_FEATURES
        if column not in df.columns
    ]

    if missing:

        print("\nMissing features:")

        for column in missing:

            print(
                " -",
                column
            )

        raise RuntimeError(
            "Required features are missing."
        )

    # --------------------------------------------------------
    # Extra columns
    # --------------------------------------------------------

    extra = [
        column
        for column in df.columns
        if column not in EXPECTED_FEATURES
    ]

    if extra:

        print("\nExtra features:")

        for column in extra:

            print(
                " -",
                column
            )

    # --------------------------------------------------------
    # Exact column order
    # --------------------------------------------------------

    df = df[
        EXPECTED_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    df = df.replace(
        [
            float("inf"),
            float("-inf")
        ],
        0
    )

    df = df.fillna(0)

    # --------------------------------------------------------
    # Remove negative values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    negative_count = int(
        (df[numeric_columns] < 0)
        .sum()
        .sum()
    )

    if negative_count > 0:

        print(
            "Negative values found:",
            negative_count
        )

        df[numeric_columns] = (
            df[numeric_columns]
            .clip(lower=0)
        )

    else:

        print(
            "Negative values: 0"
        )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    nan_count = int(
        df.isna()
        .sum()
        .sum()
    )

    print(
        "NaN values:",
        nan_count
    )

    # --------------------------------------------------------
    # Infinite
    # --------------------------------------------------------

    infinite_count = int(
        df[numeric_columns]
        .isin(
            [
                float("inf"),
                float("-inf")
            ]
        )
        .sum()
        .sum()
    )

    print(
        "Infinite values:",
        infinite_count
    )

    # --------------------------------------------------------
    # Final column check
    # --------------------------------------------------------

    if list(df.columns) != EXPECTED_FEATURES:

        raise RuntimeError(
            "Feature order does not match the model."
        )

    print(
        "Feature order: CORRECT"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Read PCAP
    # --------------------------------------------------------

    lines = read_pcap()

    if not lines:

        print(
            "\nNo packets found."
        )

        return

    # --------------------------------------------------------
    # Parse packets
    # --------------------------------------------------------

    packets = parse_packets(
        lines
    )

    if not packets:

        print(
            "\nNo TCP/UDP packets found."
        )

        return

    # --------------------------------------------------------
    # Build flows
    # --------------------------------------------------------

    flows = build_flows(
        packets
    )

    if not flows:

        print(
            "\nNo network flows generated."
        )

        return

    # --------------------------------------------------------
    # Calculate features
    # --------------------------------------------------------

    df = calculate_features(
        flows
    )

    if df.empty:

        print(
            "\nNo features generated."
        )

        return

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    df = validate_output(
        df
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FLOW EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        "Flows generated:",
        len(df)
    )

    print(
        "Features generated:",
        len(df.columns)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nFeature columns:")

    for column in df.columns:

        print(column)

    print("\n" + "=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()