
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PROJECT CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from detection.unified_detector import UnifiedNIDSDetector
from detection.live_monitor import LiveNIDSMonitor
from explainability.explainer import NIDSExplainer
from detection.recommendation_engine import RecommendationEngine

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_bool(series):
    return series.fillna(False).astype(bool)


def make_chart_layout(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=25),
        font=dict(
            family="Arial",
            color="#DCE7F5",
            size=12
        ),
        title_font=dict(
            size=16,
            color="#EAF6FF"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1")
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            font_color="white"
        )
    )
    return fig


def get_protocol_labels(df):
    if "Protocol" not in df.columns:
        return None

    protocol = pd.to_numeric(
        df["Protocol"],
        errors="coerce"
    )

    return protocol.map({
        6: "TCP",
        17: "UDP",
        1: "ICMP"
    }).fillna("Other / Unknown")


# =========================================================
# DASHBOARD
# =========================================================

def run_streamlit_dashboard():

    st.set_page_config(
        page_title="AI-NIDS | Security Operations Center",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # =====================================================
    # FUTURISTIC SOC THEME
    # =====================================================

    st.markdown("""
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(0, 180, 255, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(125, 70, 255, 0.12),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #070B16 0%,
                #0B1220 50%,
                #080D19 100%
            );

        color: #E5F0FF;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(5, 10, 20, 0.75);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0B1425,
                #101A2E
            );
        border-right: 1px solid rgba(34, 211, 238, 0.25);
    }

    [data-testid="stSidebar"] * {
        color: #DCEBFF;
    }

    h1, h2, h3 {
        color: #F0F7FF !important;
        font-weight: 800 !important;
    }

    h1 {
        letter-spacing: -1px;
    }

    p, label {
        color: #B9C9DD;
    }

    .hero-card {
        background:
            linear-gradient(
                120deg,
                rgba(15, 40, 70, 0.95),
                rgba(31, 25, 72, 0.90)
            );

        border: 1px solid rgba(34, 211, 238, 0.45);
        border-radius: 20px;
        padding: 25px 28px;
        margin-bottom: 22px;

        box-shadow:
            0 0 25px rgba(0, 190, 255, 0.10),
            inset 0 0 25px rgba(0, 190, 255, 0.04);
    }

    .hero-title {
        color: #F4FAFF;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #8EDFF5;
        font-size: 14px;
        letter-spacing: 1px;
    }

    .status-pill {
        display: inline-block;
        background: rgba(0, 220, 160, 0.12);
        color: #4FFFC2;
        border: 1px solid rgba(0, 220, 160, 0.5);
        border-radius: 30px;
        padding: 7px 14px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .section-label {
        color: #7DD3FC;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(20, 34, 57, 0.95),
                rgba(13, 23, 41, 0.95)
            );

        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 16px;
        padding: 18px 18px 15px 18px;

        box-shadow:
            0 5px 20px rgba(0, 0, 0, 0.25),
            inset 0 0 15px rgba(56, 189, 248, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #A8BED8 !important;
        font-size: 13px !important;
    }

    [data-testid="stMetricValue"] {
        color: #F1F8FF !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(56, 189, 248, 0.25);
        border-radius: 16px;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 10px;
        border: 1px solid #22D3EE;
        background: linear-gradient(
            90deg,
            #075985,
            #4338CA
        );
        color: white;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        border-color: #67E8F9;
        color: white;
        box-shadow: 0 0 18px rgba(34, 211, 238, 0.35);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(56, 189, 248, 0.20);
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown("""
<div class="hero-card">
<div class="hero-title">🛡️ AI-NIDS SECURITY OPERATIONS CENTER</div>
<div class="hero-subtitle">ARTIFICIAL INTELLIGENCE POWERED NETWORK INTRUSION DETECTION SYSTEM</div>
<br>
<span class="status-pill">● HYBRID AI DETECTION ENGINE</span>
<br><br>
<div style="color:#C6D9EE;font-size:13px;">
Random Forest &nbsp;|&nbsp;
Gradient Boosting &nbsp;|&nbsp;
Isolation Forest &nbsp;|&nbsp;
Explainable AI
</div>
</div>
""", unsafe_allow_html=True)

    # =====================================================
    # LOAD MODELS
    # =====================================================

    @st.cache_resource
    def get_detector():
        return UnifiedNIDSDetector()

    @st.cache_resource
    def get_explainer():
        return NIDSExplainer()

    @st.cache_resource
    def get_monitor():
        return LiveNIDSMonitor(
            detector=get_detector()
        )

    try:
        detector = get_detector()
        explainer = get_explainer()
        monitor = get_monitor()

    except Exception as error:
        st.error(f"Model initialization failed: {error}")
        st.stop()

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.markdown("## 🕹️ CONTROL CENTER")

    st.sidebar.caption(
        "Select your network traffic source"
    )

    mode = st.sidebar.selectbox(
        "Detection Mode",
        [
            "Latest Live Capture",
            "Live Network Monitoring",
            "Test Dataset",
            "Upload Custom CSV"
        ]
    )

    data_dir = os.path.join(
        BASE_DIR,
        "dataset",
        "processed"
    )

    live_file = os.path.join(
        BASE_DIR,
        "reports",
        "live_capture_03_v3.csv"
    )

    # =====================================================
    # LIVE MONITORING CONTROLS
    # =====================================================

    if mode == "Live Network Monitoring":

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔴 LIVE CONTROLS")

        status = monitor.get_status()

        if status["running"]:

            st.sidebar.success(
                "● Monitoring is running"
            )

            if st.sidebar.button(
                "⏹ Stop Monitoring",
                use_container_width=True
            ):
                monitor.stop()
                st.rerun()

        else:

            st.sidebar.warning(
                "● Monitoring is stopped"
            )

            if st.sidebar.button(
                "▶ Start Monitoring",
                use_container_width=True
            ):

                started = monitor.start()

                if started:
                    st.sidebar.success(
                        "Monitoring started"
                    )
                else:
                    st.sidebar.error(
                        monitor.last_error
                        or "Unable to start monitoring"
                    )

                st.rerun()

        st.sidebar.caption(
            "CICFlowMeter capture must be started separately."
        )

        st.sidebar.caption(
            "The monitor processes completed CSV snapshots."
        )

        st.sidebar.markdown("---")

        status = monitor.get_status()

        st.sidebar.write(
            "**Status:**",
            status["status"]
        )

        if status["last_processed_file"]:
            st.sidebar.write(
                "**Last file:**",
                status["last_processed_file"]
            )

        if status["last_processed_time"]:
            st.sidebar.write(
                "**Last processed:**",
                status["last_processed_time"]
            )

        if status["last_error"]:
            st.sidebar.error(
                status["last_error"]
            )

        if st.sidebar.button(
            "🔄 Refresh Dashboard",
            use_container_width=True
        ):
            st.rerun()

    # =====================================================
    # LOAD DATA
    # =====================================================

    df_input = pd.DataFrame()
    results = pd.DataFrame()

    live_monitor_mode = (
        mode == "Live Network Monitoring"
    )

    # -----------------------------------------------------
    # MODE 1: LATEST LIVE CAPTURE
    # -----------------------------------------------------

    if mode == "Latest Live Capture":

        if os.path.exists(live_file):

            try:
                df_input = pd.read_csv(live_file)

                st.sidebar.success(
                    f"Loaded {len(df_input)} live flows"
                )

                st.sidebar.caption(
                    "Source: reports/live_capture_03_v3.csv"
                )

            except Exception as error:
                st.error(
                    f"Unable to load live capture: {error}"
                )

        else:
            st.sidebar.error(
                "Live capture file not found."
            )

    # -----------------------------------------------------
    # MODE 2: LIVE NETWORK MONITORING
    # -----------------------------------------------------

    elif live_monitor_mode:

        status = monitor.get_status()

        st.markdown(
            '<div class="section-label">LIVE NETWORK TELEMETRY</div>',
            unsafe_allow_html=True
        )

        st.subheader("🔴 Live Network Monitoring")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Monitoring Status",
            "RUNNING" if status["running"] else "STOPPED"
        )

        c2.metric(
            "Last Processed File",
            status["last_processed_file"] or "None"
        )

        c3.metric(
            "Last Processing Time",
            status["last_processed_time"] or "Not available"
        )

        if status["last_error"]:
            st.error(status["last_error"])

        if not status["running"]:
            st.info(
                "Start monitoring from the sidebar to process "
                "new network captures."
            )

        latest = status["latest_results"]

        if latest.empty:
            st.warning(
                "No new capture has been processed yet. "
                "Start CICFlowMeter and wait for a stable CSV snapshot."
            )
            st.stop()

        result_columns = [
            "flow_index",
            "protocol",
            "verdict",
            "severity",
            "is_attack",
            "is_unknown",
            "model_is_attack",
            "attack_probability",
            "benign_probability",
            "predicted_category",
            "category_confidence",
            "zero_day_anomaly",
            "anomaly_score"
        ]

        available_result_columns = [
            column for column in result_columns
            if column in latest.columns
        ]

        results = latest[
            available_result_columns
        ].copy()

        df_input = latest.drop(
            columns=available_result_columns,
            errors="ignore"
        ).copy()

        st.success(
            f"Loaded {len(results)} newly processed flows"
        )

    # -----------------------------------------------------
    # MODE 3: TEST DATASET
    # -----------------------------------------------------

    elif mode == "Test Dataset":

        possible_paths = [
            os.path.join(
                data_dir,
                "prepared_v3",
                "X_test.csv"
            ),
            os.path.join(
                data_dir,
                "prepared",
                "X_test.csv"
            )
        ]

        test_file = next(
            (
                path for path in possible_paths
                if os.path.exists(path)
            ),
            None
        )

        if test_file:

            try:
                test_df = pd.read_csv(test_file)

                max_flows = min(
                    500,
                    len(test_df)
                )

                if max_flows > 0:

                    num_flows = st.sidebar.slider(
                        "Number of flows",
                        min_value=min(10, max_flows),
                        max_value=max_flows,
                        value=min(50, max_flows),
                        step=10 if max_flows >= 10 else 1
                    )

                    df_input = test_df.head(num_flows)

                    st.sidebar.success(
                        f"Loaded {len(df_input)} test flows"
                    )

            except Exception as error:
                st.error(
                    f"Unable to load test dataset: {error}"
                )

        else:
            st.sidebar.error(
                "X_test.csv was not found."
            )

    # -----------------------------------------------------
    # MODE 4: CUSTOM CSV
    # -----------------------------------------------------

    elif mode == "Upload Custom CSV":

        uploaded_file = st.sidebar.file_uploader(
            "Upload Network Flow CSV",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:
                df_input = pd.read_csv(uploaded_file)

                st.sidebar.success(
                    f"Uploaded {len(df_input)} flows"
                )

            except Exception as error:
                st.error(
                    f"Unable to read CSV: {error}"
                )

    # =====================================================
    # VALIDATE INPUT
    # =====================================================

    if df_input.empty:
        st.warning(
          "No network flows loaded. "
          "Select a valid traffic source."
        )
        st.stop()

    # =====================================================
    # RUN DETECTION
    # =====================================================

    if not live_monitor_mode:

        with st.spinner(
            "⚡ AI engine analyzing network traffic..."
        ):

            try:
                results = detector.predict(df_input)

            except Exception as error:
                st.error(
                    f"Detection failed: {error}"
                )
                st.stop()

    if results.empty:
        st.warning(
            "No prediction results are available."
        )
        st.stop()

    df_input = df_input.reset_index(drop=True)
    results = results.reset_index(drop=True)

    # =====================================================
    # CALCULATE SECURITY METRICS
    # =====================================================

    total_flows = len(results)

    attack_mask = safe_bool(results["is_attack"])
    unknown_mask = safe_bool(results["is_unknown"])

    attack_count = int(attack_mask.sum())
    unknown_count = int(unknown_mask.sum())

    benign_count = max(
        0,
        total_flows - attack_count - unknown_count
    )

    anomaly_mask = safe_bool(
        results["zero_day_anomaly"]
    )

    zero_day_count = int(
        (
            anomaly_mask
            & ~attack_mask
            & ~unknown_mask
        ).sum()
    )

    attack_percentage = (
        attack_count / total_flows * 100
    )

    benign_percentage = (
        benign_count / total_flows * 100
    )

    unknown_percentage = (
        unknown_count / total_flows * 100
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

    st.markdown(
        '<div class="section-label">SYSTEM OVERVIEW / SECURITY TELEMETRY</div>',
        unsafe_allow_html=True
    )

    st.subheader("📊 Network Security Overview")

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "🌐 Total Flows",
        f"{total_flows:,}"
    )

    k2.metric(
        "🚨 Predicted Attacks",
        f"{attack_count:,}",
        f"{attack_percentage:.1f}%",
        delta_color="inverse"
    )

    k3.metric(
        "🟢 Benign",
        f"{benign_count:,}",
        f"{benign_percentage:.1f}%"
    )

    k4.metric(
        "⚠️ Unknown",
        f"{unknown_count:,}",
        f"{unknown_percentage:.1f}%"
    )

    k5.metric(
        "🧬 Anomaly Flags",
        f"{zero_day_count:,}"
    )

    st.caption(
        "Attack and benign figures are model predictions, "
        "not independently confirmed incidents. "
        "Unknown flows are not counted as attacks."
    )

    st.divider()

    # =====================================================
    # INTERACTIVE CHARTS
    # =====================================================

    st.markdown(
        '<div class="section-label">THREAT INTELLIGENCE / VISUAL ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.subheader("📡 Network Threat Intelligence")

    # -----------------------------------------------------
    # DONUT CHART: THREAT DISTRIBUTION
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown("#### 🎯 Traffic Classification")

        distribution = pd.DataFrame({
            "Traffic": [
                "Predicted Attack",
                "Benign",
                "Unknown"
            ],
            "Flows": [
                attack_count,
                benign_count,
                unknown_count
            ]
        })

        fig_donut = px.pie(
            distribution,
            names="Traffic",
            values="Flows",
            hole=0.68,
            color="Traffic",
            color_discrete_map={
                "Predicted Attack": "#FF4560",
                "Benign": "#00D6A0",
                "Unknown": "#F5A623"
            }
        )

        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent+label",
            marker=dict(
                line=dict(
                    color="#0B1220",
                    width=3
                )
            )
        )

        fig_donut.add_annotation(
            text=f"<b>{total_flows:,}</b><br>FLOWS",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=19,
                color="#FFFFFF"
            )
        )

        make_chart_layout(fig_donut, 370)

        st.plotly_chart(
            fig_donut,
            use_container_width=True
        )

    # -----------------------------------------------------
    # PROTOCOL DONUT CHART
    # -----------------------------------------------------

    with right:

        st.markdown("#### 🌐 Protocol Distribution")

        protocol_labels = get_protocol_labels(df_input)

        if protocol_labels is not None:

            protocol_counts = (
                protocol_labels.value_counts()
                .rename_axis("Protocol")
                .reset_index(name="Flows")
            )

            fig_protocol = px.pie(
                protocol_counts,
                names="Protocol",
                values="Flows",
                hole=0.58,
                color_discrete_sequence=[
                    "#22D3EE",
                    "#A78BFA",
                    "#34D399",
                    "#F59E0B"
                ]
            )

            fig_protocol.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(
                    line=dict(
                        color="#0B1220",
                        width=3
                    )
                )
            )

            make_chart_layout(fig_protocol, 370)

            st.plotly_chart(
                fig_protocol,
                use_container_width=True
            )

        else:

            st.info(
                "Protocol information is unavailable."
            )

    # =====================================================
    # ATTACK PROBABILITY LINE GRAPH
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-label">MODEL CONFIDENCE / FLOW ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.subheader("📈 Attack Probability Across Flow Sequence")

    if "attack_probability" in results.columns:

        probability = pd.to_numeric(
            results["attack_probability"],
            errors="coerce"
        ).fillna(0)

        line_df = pd.DataFrame({
            "Flow Sequence": range(1, len(results) + 1),
            "Attack Probability": probability
        })

        fig_line = px.line(
            line_df,
            x="Flow Sequence",
            y="Attack Probability",
            markers=True
        )

        fig_line.update_traces(
            line=dict(
                color="#22D3EE",
                width=3
            ),
            marker=dict(
                color="#A78BFA",
                size=6
            ),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.08)"
        )

        fig_line.update_layout(
            xaxis_title="Flow Sequence",
            yaxis_title="Attack Probability (%)",
            yaxis=dict(
                rangemode="tozero"
            )
        )

        make_chart_layout(fig_line, 360)

        st.plotly_chart(
            fig_line,
            use_container_width=True
        )

        st.caption(
            "This graph follows the order of analyzed flows. "
            "It is not a timestamp-based traffic timeline."
        )

    else:

        st.info(
            "Attack probability information is unavailable."
        )

    # =====================================================
    # ATTACK CATEGORIES AND SEVERITY
    # =====================================================

    st.divider()

    st.subheader("🚨 Threat Classification Analytics")

    col_cat, col_sev = st.columns(2)

    attack_results = results[
        attack_mask & ~unknown_mask
    ].copy()

    with col_cat:

        st.markdown("#### ⚔️ Predicted Attack Categories")

        if (
            not attack_results.empty
            and "predicted_category" in attack_results.columns
        ):

            category_counts = (
                attack_results["predicted_category"]
                .fillna("Unclassified")
                .astype(str)
                .value_counts()
                .reset_index()
            )

            category_counts.columns = [
                "Attack Category",
                "Flows"
            ]

            fig_category = px.bar(
                category_counts,
                x="Flows",
                y="Attack Category",
                orientation="h",
                color="Flows",
                color_continuous_scale=[
                    "#312E81",
                    "#2563EB",
                    "#22D3EE"
                ]
            )

            fig_category.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(
                    autorange="reversed"
                ),
                xaxis_title="Number of Flows",
                yaxis_title=""
            )

            make_chart_layout(
                fig_category,
                max(300, 65 * len(category_counts))
            )

            st.plotly_chart(
                fig_category,
                use_container_width=True
            )

        else:

            st.info(
                "No attack predictions in this dataset."
            )

    with col_sev:

        st.markdown("#### 🔥 Severity Distribution")

        if "severity" in results.columns:

            severity_counts = (
                results["severity"]
                .fillna("UNKNOWN")
                .astype(str)
                .value_counts()
                .reset_index()
            )

            severity_counts.columns = [
                "Severity",
                "Flows"
            ]

            severity_colors = {
                "CRITICAL": "#FF1744",
                "HIGH": "#FF7043",
                "MEDIUM": "#F5A623",
                "LOW": "#22D3EE",
                "NORMAL": "#00D6A0",
                "UNKNOWN": "#A78BFA"
            }

            fig_severity = px.bar(
                severity_counts,
                x="Severity",
                y="Flows",
                color="Severity",
                color_discrete_map=severity_colors
            )

            fig_severity.update_layout(
                showlegend=False,
                xaxis_title="Severity Level",
                yaxis_title="Number of Flows"
            )

            make_chart_layout(fig_severity, 350)

            st.plotly_chart(
                fig_severity,
                use_container_width=True
            )

        else:

            st.info(
                "Severity information is unavailable."
            )

    # =====================================================
    # SECURITY DETECTION FEED
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-label">THREAT MONITORING / DETECTION LOGS</div>',
        unsafe_allow_html=True
    )

    st.subheader("🔍 Network Security Detection Feed")

    display_cols = [
        "flow_index",
        "protocol",
        "verdict",
        "severity",
        "attack_probability",
        "predicted_category",
        "category_confidence",
        "zero_day_anomaly",
        "anomaly_score"
    ]

    available_cols = [
        column for column in display_cols
        if column in results.columns
    ]

    def color_severity(value):

        if value == "CRITICAL":
            return (
                "background-color: #8B0000; "
                "color: white"
            )

        if value == "HIGH":
            return (
                "background-color: #D2691E; "
                "color: white"
            )

        if value in ["SUSPICIOUS", "UNKNOWN", "MEDIUM"]:
            return (
                "background-color: #765000; "
                "color: white"
            )

        if value in ["NORMAL", "LOW"]:
            return (
                "background-color: #176B45; "
                "color: white"
            )

        return ""

    if available_cols:

        styled = results[available_cols].style

        if "severity" in available_cols:

            if hasattr(styled, "map"):
                styled = styled.map(
                    color_severity,
                    subset=["severity"]
                )
            else:
                styled = styled.applymap(
                    color_severity,
                    subset=["severity"]
                )

        st.dataframe(
            styled,
            use_container_width=True,
            height=420
        )

    # =====================================================
    # EXPORT REPORT
    # =====================================================

    st.divider()

    st.subheader("📥 Export Detection Report")

    export_df = pd.concat(
        [
            df_input.reset_index(drop=True),
            results.reset_index(drop=True)
        ],
        axis=1
    )

    export_df = export_df.loc[
        :,
        ~export_df.columns.duplicated()
    ]

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Complete Detection Report",
        data=csv_data,
        file_name="AI_NIDS_Detection_Report.csv",
        mime="text/csv",
        use_container_width=True
    )

    # =====================================================
    # DETAILED FLOW INSPECTION
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-label">INVESTIGATION / FLOW FORENSICS</div>',
        unsafe_allow_html=True
    )

    st.subheader("🔬 Detailed Network Flow Inspection")

    flow_positions = list(range(len(results)))

    selected_position = st.selectbox(
        "Select a flow to investigate",
        flow_positions,
        format_func=lambda x: (
            f"Flow {results.iloc[x].get('flow_index', x)}"
        )
    )

    flow_row = results.iloc[selected_position]
    flow_features = df_input.iloc[selected_position]

    f1, f2, f3 = st.columns(3)

    with f1:

        st.metric(
            "Detection Verdict",
            str(flow_row.get("verdict", "N/A"))
        )

        st.metric(
            "Severity",
            str(flow_row.get("severity", "N/A"))
        )

    with f2:

        st.metric(
            "Attack Probability",
            f"{flow_row.get('attack_probability', 0)}%"
        )

        st.metric(
            "Predicted Category",
            str(flow_row.get("predicted_category", "N/A"))
        )

    with f3:

        st.metric(
            "Anomaly Score",
            str(flow_row.get("anomaly_score", "N/A"))
        )

        st.metric(
            "Zero-Day Anomaly",
            "YES" if bool(
                flow_row.get("zero_day_anomaly", False)
            ) else "NO"
        )

    if "Protocol" in df_input.columns:

        st.markdown(
            f"**Protocol:** `{flow_features['Protocol']}`"
        )

    # =====================================================
    # SECURITY RECOMMENDATIONS
    # =====================================================

    st.divider()

    st.subheader("🛡️ Security Recommendations")

    flow_is_attack = bool(
        flow_row.get("is_attack", False)
    )

    flow_is_unknown = bool(
        flow_row.get("is_unknown", False)
    )

    flow_is_anomaly = bool(
        flow_row.get("zero_day_anomaly", False)
    )

    predicted_category = str(
        flow_row.get("predicted_category", "UNKNOWN")
    )

    category_is_uncertain = (
        predicted_category.strip().upper()
        in ["BENIGN", "UNKNOWN", "NONE", ""]
    )

    if flow_is_attack and not flow_is_unknown:

        if category_is_uncertain:

            st.warning(
                "⚠️ Potential attack detected, "
                "but the attack category is uncertain."
            )

            st.markdown(
                "**Recommendation:** Review this flow manually "
                "and validate its network behavior."
            )

        else:

            try:

                recommendation = (
                    RecommendationEngine.get_recommendation(
                        predicted_category
                    )
                )

                st.error(
                    f"⚠️ Predicted {predicted_category} Attack"
                )

                st.markdown(
                    f"**Severity:** "
                    f"{recommendation['severity']}"
                )

                st.markdown(
                    f"**Description:** "
                    f"{recommendation['description']}"
                )

                st.markdown(
                    f"**Potential Risk:** "
                    f"{recommendation['risk']}"
                )

                st.markdown(
                    "### Recommended Defensive Actions"
                )

                for index, action in enumerate(
                    recommendation["actions"],
                    start=1
                ):
                    st.markdown(
                        f"{index}. {action}"
                    )

            except Exception as error:

                st.warning(
                    f"Recommendation unavailable: {error}"
                )

        st.info(
            "Recommendations only. "
            "No automatic firewall or network changes are performed."
        )

    elif flow_is_unknown:

        st.warning(
            "This flow uses an unsupported or missing protocol. "
            "It is marked UNKNOWN and is not classified as benign or attack."
        )

    elif flow_is_anomaly:

        st.warning(
            "An anomaly has been flagged. "
            "Review this flow for unusual behavior."
        )

    else:

        st.success(
            "No attack-specific recommendations are needed "
            "for this flow because it was predicted as benign."
        )

    # =====================================================
    # EXPLAINABLE AI
    # =====================================================

    st.divider()

    st.subheader("🧠 AI Threat Reasoning (XAI)")

    if (
        flow_is_attack or flow_is_anomaly
    ) and not flow_is_unknown:

        try:

            reasons = explainer.explain_flow(
                flow_features
            )

            if reasons:

                for reason in reasons:

                    st.info(
                        f"**{reason['feature']}**: "
                        f"{reason['reason']} "
                        f"(Observed value: "
                        f"`{reason['observed_value']}`)"
                    )

            else:

                st.info(
                    "No explanation available for this flow."
                )

        except Exception as error:

            st.warning(
                f"XAI explanation unavailable: {error}"
            )

    elif flow_is_unknown:

        st.warning(
            "XAI explanation is unavailable for this "
            "unsupported or unknown flow."
        )

    else:

        st.success(
            "The model classified this flow as benign."
        )

    # =====================================================
    # OBSERVED NETWORK FEATURES
    # =====================================================

    st.divider()

    st.subheader("📐 Observed Network Flow Features")

    key_features = [
        "Destination Port",
        "Flow Duration",
        "Flow Packets/s",
        "Flow Bytes/s",
        "Total Fwd Packets",
        "Total Backward Packets",
        "SYN Flag Count",
        "ACK Flag Count"
    ]

    available_features = [
        feature for feature in key_features
        if feature in flow_features.index
    ]

    if available_features:

        feature_table = pd.DataFrame(
            [
                flow_features[
                    available_features
                ].to_dict()
            ],
            index=["Observed Value"]
        )

        st.dataframe(
            feature_table,
            use_container_width=True
        )

    # =====================================================
    # FOOTER
    # =====================================================

    st.divider()

    st.markdown("""
    <div style="
        text-align:center;
        padding:20px;
        border-top:1px solid rgba(34,211,238,0.2);
    ">
        <div style="
            color:#67E8F9;
            font-size:18px;
            font-weight:800;
        ">
            🛡️ AI-NIDS SECURITY OPERATIONS CENTER
        </div>

        <div style="
            color:#94A3B8;
            font-size:12px;
            margin-top:8px;
        ">
            Random Forest + Gradient Boosting +
            Isolation Forest + Explainable AI
        </div>

        <div style="
            color:#64748B;
            font-size:11px;
            margin-top:12px;
        ">
            Academic AI-NIDS Prototype |
            Network Threat Detection and Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Predictions require independent validation "
        "before being treated as confirmed security incidents."
    )

    # =====================================================
    # LIVE MONITORING REFRESH NOTE
    # =====================================================

    if live_monitor_mode:

        status = monitor.get_status()

        if status["running"]:

            st.info(
                "Live monitoring is active. "
                "Use Refresh Dashboard in the sidebar "
                "to view the latest processed capture."
            )


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":

    if HAS_STREAMLIT:

        run_streamlit_dashboard()

    else:

        print(
            "Streamlit is not installed. "
            "Install it using: "
            "python -m pip install streamlit"
        )
