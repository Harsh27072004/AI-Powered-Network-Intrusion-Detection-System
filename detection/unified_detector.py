import os
import joblib
import pandas as pd
import numpy as np


class UnifiedNIDSDetector:
    """
    Hybrid AI-NIDS detector.

    V3 Random Forest:
        Primary BENIGN / ATTACK classification.

    Gradient Boosting:
        Secondary attack-category prediction.

    Isolation Forest:
        Anomaly scoring.

    Protocol handling:
        TCP (6) and UDP (17) are supported.
        Other protocols are marked UNKNOWN.
    """

    def __init__(self, model_dir=None):

        if model_dir is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            model_dir = os.path.join(base_dir, "models")

        self.model_dir = model_dir

        print(f"Loading AI-NIDS models from: {self.model_dir}")

        # LOAD V3 BINARY MODEL
        self.rf_binary = joblib.load(
            os.path.join(
                model_dir,
                "random_forest_binary_v3.pkl"
            )
        )

        threshold_path = os.path.join(
            model_dir,
            "random_forest_binary_v3_threshold.txt"
        )

        with open(threshold_path, "r", encoding="utf-8") as file:
            self.threshold = float(file.read().strip())

        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid classification threshold.")

        self.feature_names = list(
            self.rf_binary.feature_names_in_
        )

        if len(self.feature_names) != 61:
            raise ValueError(
                f"Expected 61 features, got {len(self.feature_names)}"
            )

        # LOAD MULTICLASS MODEL
        self.hgb_multiclass = joblib.load(
            os.path.join(
                model_dir,
                "gradient_boost_multiclass.pkl"
            )
        )

        self.multiclass_features = list(
            self.hgb_multiclass.feature_names_in_
        )

        self.class_labels = joblib.load(
            os.path.join(model_dir, "class_labels.pkl")
        )

        # LOAD ISOLATION FOREST
        self.iso_anomaly = joblib.load(
            os.path.join(
                model_dir,
                "isolation_forest_anomaly.pkl"
            )
        )

        print("V3 binary model loaded.")
        print("Multiclass model loaded.")
        print("Isolation Forest loaded.")
        print("Threshold:", self.threshold)

    # =====================================================
    # FEATURE PREPROCESSING
    # =====================================================

    def _prepare_features(self, df, feature_names):

        missing = [
            feature for feature in feature_names
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing)
            )

        X = df[feature_names].copy()

        for column in X.columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if X.isna().any().any():

            bad_columns = X.columns[
                X.isna().any()
            ].tolist()

            raise ValueError(
                "Missing or invalid values in: "
                + ", ".join(bad_columns)
            )

        if not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():

            raise ValueError(
                "Input contains non-finite values."
            )

        return X

    def preprocess_flows(self, df_flows):

        if not isinstance(df_flows, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if df_flows.empty:
            raise ValueError(
                "No network flows supplied."
            )

        df = df_flows.copy()

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        if df.columns.duplicated().any():
            raise ValueError(
                "Duplicate feature column names detected."
            )

        return self._prepare_features(
            df,
            self.feature_names
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(self, df_flows):

        if not isinstance(df_flows, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if df_flows.empty:
            raise ValueError("No network flows supplied.")

        # Normalize column names once.
        df = df_flows.copy()

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        if df.columns.duplicated().any():
            raise ValueError(
                "Duplicate column names detected."
            )

        # Preserve protocol as metadata.
        # It is NOT passed to the trained models.
        protocol_available = "Protocol" in df.columns

        if protocol_available:

            protocol_values = pd.to_numeric(
                df["Protocol"],
                errors="coerce"
            )

            supported_protocols = protocol_values.isin([6, 17])

        else:

            # Prepared CIC-IDS2017 test data has no Protocol column.
            # Preserve normal model behaviour for that data.
            protocol_values = pd.Series(
                np.nan,
                index=df.index
            )

            supported_protocols = pd.Series(
                True,
                index=df.index
            )

        # Prepare exactly 61 binary model features.
        X = self.preprocess_flows(df)

        # -------------------------------------------------
        # BINARY RANDOM FOREST PREDICTION
        # -------------------------------------------------

        probabilities = self.rf_binary.predict_proba(X)

        if probabilities.shape[1] != 2:
            raise ValueError(
                "Binary model must return two class probabilities."
            )

        classes = list(self.rf_binary.classes_)

        if 1 not in classes or 0 not in classes:
            raise ValueError(
                "Binary model must contain classes 0 and 1."
            )

        attack_column = classes.index(1)

        attack_probabilities = probabilities[:, attack_column]

        if not np.isfinite(attack_probabilities).all():
            raise ValueError(
                "Invalid binary model probabilities."
            )

        if (
            (attack_probabilities < 0).any()
            or (attack_probabilities > 1).any()
        ):
            raise ValueError(
                "Binary probabilities are outside valid range."
            )

        binary_predictions = (
            attack_probabilities >= self.threshold
        ).astype(int)

        # -------------------------------------------------
        # MULTICLASS PREDICTION
        # -------------------------------------------------

        multiclass_available = all(
            feature in df.columns
            for feature in self.multiclass_features
        )

        categories = ["NOT_AVAILABLE"] * len(X)

        category_confidences = np.zeros(len(X))

        if multiclass_available:

            try:
                X_multi = self._prepare_features(
                    df,
                    self.multiclass_features
                )

                multi_predictions = (
                    self.hgb_multiclass.predict(X_multi)
                )

                multi_probabilities = (
                    self.hgb_multiclass.predict_proba(X_multi)
                )

                if not np.isfinite(
                    multi_probabilities
                ).all():
                    raise ValueError(
                        "Invalid multiclass probabilities."
                    )

                if multi_probabilities.shape[0] != len(X):
                    raise ValueError(
                        "Multiclass prediction row mismatch."
                    )

                categories = [
                    self.class_labels.get(
                        int(prediction),
                        "UNKNOWN"
                    )
                    for prediction in multi_predictions
                ]

                category_confidences = np.max(
                    multi_probabilities,
                    axis=1
                )

            except (ValueError, TypeError):
                categories = ["NOT_AVAILABLE"] * len(X)
                category_confidences = np.zeros(len(X))

        # -------------------------------------------------
        # ISOLATION FOREST
        # -------------------------------------------------

        iso_scores = self.iso_anomaly.decision_function(X)

        iso_predictions = self.iso_anomaly.predict(X)

        anomaly_flags = iso_predictions == -1

        # -------------------------------------------------
        # FINAL VERDICTS
        # -------------------------------------------------

        results = []

        for i in range(len(X)):

            model_is_attack = bool(
                binary_predictions[i] == 1
            )

            attack_prob = float(
                attack_probabilities[i]
            )

            benign_prob = 1.0 - attack_prob

            anomaly_flag = bool(
                anomaly_flags[i]
            )

            category = categories[i]

            # Determine whether protocol is supported.
            is_unknown = not bool(
                supported_protocols.iloc[i]
            )

            if is_unknown:

                verdict = "UNKNOWN (UNSUPPORTED_PROTOCOL)"
                severity = "UNKNOWN"
                is_attack = False

            else:

                is_attack = model_is_attack

                if not is_attack and not anomaly_flag:

                    severity = "NORMAL"
                    verdict = "BENIGN"

                elif not is_attack and anomaly_flag:

                    severity = "SUSPICIOUS"
                    verdict = "ANOMALOUS_BENIGN_PREDICTION"

                else:

                    if category in ["DDoS", "DoS"]:
                        severity = "CRITICAL"

                    elif category in ["BruteForce", "WebAttack"]:
                        severity = "HIGH"

                    else:
                        severity = "MEDIUM"

                    if category == "BENIGN":
                        verdict = "ATTACK (CATEGORY_UNCERTAIN)"

                    elif category == "NOT_AVAILABLE":
                        verdict = "ATTACK (CATEGORY_UNAVAILABLE)"

                    elif category == "UNKNOWN":
                        verdict = "ATTACK (CATEGORY_UNKNOWN)"

                    else:
                        verdict = f"ATTACK ({category})"

            results.append({

                "flow_index": i,

                "verdict": verdict,

                "severity": severity,

                "is_attack": is_attack,

                "is_unknown": is_unknown,

                "model_is_attack": model_is_attack,

                "protocol": (
                    protocol_values.iloc[i]
                    if protocol_available
                    else np.nan
                ),

                "attack_probability": round(
                    attack_prob * 100, 2
                ),

                "benign_probability": round(
                    benign_prob * 100, 2
                ),

                "predicted_category": category,

                "category_confidence": round(
                    float(category_confidences[i]) * 100,
                    2
                ),

                "zero_day_anomaly": anomaly_flag,

                "anomaly_score": round(
                    float(iso_scores[i]), 4
                )
            })

        return pd.DataFrame(results)


# =========================================================
# INITIALIZATION TEST
# =========================================================

if __name__ == "__main__":

    detector = UnifiedNIDSDetector()

    print("\nUnified V3 detector initialized successfully.")

    print(
        "Expected binary features:",
        len(detector.feature_names)
    )

    print(
        "Multiclass features:",
        len(detector.multiclass_features)
    )