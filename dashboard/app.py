
import os
import sys
import pandas as pd
import numpy as np

# =========================================================
# PROJECT CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from detection.unified_detector import UnifiedNIDSDetector
from explainability.explainer import NIDSExplainer

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# =========================================================
# DASHBOARD
# =========================================================

def run_streamlit_dashboard():

    st.set_page_config(
        page_title="AI-NIDS SOC Console",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -----------------------------------------------------
    # CUSTOM STYLING
    # -----------------------------------------------------

    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }

    .attack-badge {
        color: #FF4B4B;
        font-weight: bold;
    }

    .benign-badge {
        color: #00CC96;
        font-weight: bold;
    }

    .unknown-badge {
        color: #FFA500;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.title(
        "🛡️ AI-NIDS: AI-Powered Network Intrusion Detection System"
    )

    st.markdown(
        "**Hybrid AI Threat Detection Engine** | "
        "Random Forest + Gradient Boosting + "
        "Isolation Forest + Explainable AI"
    )

    st.caption(
        "Network traffic analysis and intrusion detection console"
    )

    # -----------------------------------------------------
    # LOAD MODELS
    # -----------------------------------------------------

    @st.cache_resource
    def get_detector():
        return UnifiedNIDSDetector()

    @st.cache_resource
    def get_explainer():
        return NIDSExplainer()

    try:
        detector = get_detector()
        explainer = get_explainer()

    except Exception as error:
        st.error(f"Model initialization failed: {error}")
        st.stop()

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    st.sidebar.header("🕹️ Detection Control Panel")

    mode = st.sidebar.selectbox(
        "Select Traffic Source",
        [
            "Latest Live Capture",
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

    # -----------------------------------------------------
    # LOAD SELECTED DATASET
    # -----------------------------------------------------

    df_input = pd.DataFrame()

    if mode == "Latest Live Capture":

        if os.path.exists(live_file):

            df_input = pd.read_csv(live_file)

            st.sidebar.success(
                f"Loaded {len(df_input)} live flows"
            )

            st.sidebar.caption(
                "Source: reports/live_capture_03_v3.csv"
            )

        else:

            st.sidebar.error(
                "Live capture file not found."
            )

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

                num_flows = st.sidebar.slider(
                    "Number of flows to inspect",
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

    else:

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

    # -----------------------------------------------------
    # VALIDATE INPUT
    # -----------------------------------------------------

    if df_input.empty:

        st.warning(
            "No network flows loaded. "
            "Select a valid traffic source."
        )

        st.stop()

    # -----------------------------------------------------
    # RUN DETECTION
    # -----------------------------------------------------

    with st.spinner(
        "Analyzing network traffic with AI-NIDS..."
    ):

        try:
            results = detector.predict(df_input)

        except Exception as error:

            st.error(
                f"Detection failed: {error}"
            )

            st.stop()

    # -----------------------------------------------------
    # CALCULATE METRICS
    # -----------------------------------------------------

    total_flows = len(results)

    attack_count = int(
        results["is_attack"].sum()
    )

    unknown_count = int(
        results["is_unknown"].sum()
    )

    benign_count = (
        total_flows
        - attack_count
        - unknown_count
    )

    zero_day_count = int(
        (
            results["zero_day_anomaly"]
            & ~results["is_attack"]
            & ~results["is_unknown"]
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

    # -----------------------------------------------------
    # TOP KPI METRICS
    # -----------------------------------------------------

    st.subheader("📊 Network Security Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Flows",
        f"{total_flows:,}"
    )

    col2.metric(
        "Predicted Attacks",
        f"{attack_count:,}",
        f"{attack_percentage:.1f}%",
        delta_color="inverse"
    )

    col3.metric(
        "Benign",
        f"{benign_count:,}",
        f"{benign_percentage:.1f}%"
    )

    col4.metric(
        "Unknown",
        f"{unknown_count:,}",
        f"{unknown_percentage:.1f}%"
    )

    col5.metric(
        "Anomaly Flags",
        f"{zero_day_count:,}"
    )

    st.caption(
        "Attack and benign figures are model predictions, "
        "not independently confirmed incidents."
    )

    st.divider()

    # -----------------------------------------------------
    # THREAT DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("📈 Threat Distribution")

    col_chart, col_protocol = st.columns(2)

    with col_chart:

        distribution = pd.DataFrame({
            "Traffic Type": [
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

        st.bar_chart(
            distribution.set_index("Traffic Type")
        )

    with col_protocol:

        st.markdown("#### Network Protocol Distribution")

        if "Protocol" in df_input.columns:

            protocol_counts = (
                pd.to_numeric(
                    df_input["Protocol"],
                    errors="coerce"
                )
                .map({
                    6: "TCP (6)",
                    17: "UDP (17)"
                })
                .fillna("Other / Unknown")
                .value_counts()
            )

            st.bar_chart(protocol_counts)

        else:

            st.info(
                "Protocol information is unavailable "
                "for this dataset."
            )

    st.divider()

    # -----------------------------------------------------
    # ATTACK CATEGORY DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("🚨 Predicted Attack Categories")

    attack_results = results[
        results["is_attack"]
        & ~results["is_unknown"]
    ]

    if not attack_results.empty:

        category_counts = (
            attack_results[
                "predicted_category"
            ]
            .value_counts()
        )

        st.bar_chart(category_counts)

    else:

        st.info(
            "No attack predictions in this dataset."
        )

    st.divider()

    # -----------------------------------------------------
    # SECURITY INCIDENT FEED
    # -----------------------------------------------------

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

        elif value == "HIGH":
            return (
                "background-color: #D2691E; "
                "color: white"
            )

        elif value in ["SUSPICIOUS", "UNKNOWN"]:
            return (
                "background-color: #B8860B; "
                "color: white"
            )

        elif value == "NORMAL":
            return (
                "background-color: #176B45; "
                "color: white"
            )

        return ""

    styled = results[
        available_cols
    ].style

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
        height=400
    )

    # -----------------------------------------------------
    # EXPORT RESULTS
    # -----------------------------------------------------

    st.subheader("📥 Export Detection Results")

    export_df = pd.concat(
        [
            df_input.reset_index(drop=True),
            results.reset_index(drop=True)
        ],
        axis=1
    )

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Complete Detection Report",
        data=csv_data,
        file_name="AI_NIDS_Detection_Report.csv",
        mime="text/csv"
    )

    st.divider()

    # -----------------------------------------------------
    # DETAILED FLOW INSPECTION
    # -----------------------------------------------------

    st.subheader(
        "🔬 Detailed Network Flow Inspection"
    )

    selected_idx = st.selectbox(
        "Select Flow Index",
        results["flow_index"].tolist()
    )

    flow_row = results.iloc[
        int(selected_idx)
    ]

    flow_features = df_input.iloc[
        int(selected_idx)
    ]

    f1, f2, f3 = st.columns(3)

    with f1:

        st.markdown(
            f"**Verdict:** `{flow_row['verdict']}`"
        )

        st.markdown(
            f"**Severity:** `{flow_row['severity']}`"
        )

    with f2:

        st.markdown(
            f"**Attack Probability:** "
            f"`{flow_row['attack_probability']}%`"
        )

        st.markdown(
            f"**Predicted Category:** "
            f"`{flow_row['predicted_category']}`"
        )

    with f3:

        st.markdown(
            f"**Anomaly Score:** "
            f"`{flow_row['anomaly_score']}`"
        )

        st.markdown(
            f"**Anomaly Flag:** "
            f"`{'YES' if flow_row['zero_day_anomaly'] else 'NO'}`"
        )

    if "Protocol" in df_input.columns:

        st.markdown(
            f"**Protocol:** `{flow_row['protocol']}`"
        )

    # -----------------------------------------------------
    # EXPLAINABLE AI
    # -----------------------------------------------------

    if (
        flow_row["is_attack"]
        or flow_row["zero_day_anomaly"]
    ) and not flow_row["is_unknown"]:

        st.markdown(
            "#### 🧠 AI Threat Reasoning (XAI)"
        )

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

    elif flow_row["is_unknown"]:

        st.warning(
            "This flow uses an unsupported or missing "
            "protocol. It is marked UNKNOWN and is not "
            "classified as benign or attack."
        )

    else:

        st.success(
            "The model classified this flow as benign."
        )

    # -----------------------------------------------------
    # OBSERVED FLOW FEATURES
    # -----------------------------------------------------

    st.markdown(
        "#### 📐 Observed Network Flow Features"
    )

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

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.divider()

    st.caption(
        "AI-NIDS | Hybrid Network Intrusion Detection "
        "System | Random Forest + Gradient Boosting + "
        "Isolation Forest + Explainable AI"
    )

    st.caption(
        "This dashboard performs batch analysis of captured "
        "network flows. Predictions require independent "
        "validation before being treated as confirmed incidents."
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
            "Install it using: python -m pip install streamlit"
        )
