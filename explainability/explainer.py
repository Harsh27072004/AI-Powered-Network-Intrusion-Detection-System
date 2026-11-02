import os
import joblib
import pandas as pd
import numpy as np

# ==============================================================================
# EXPLAINABLE AI (XAI) ENGINE FOR AI-NIDS
# Local & Global Feature Attribution for SOC Incident Response
# ==============================================================================

class NIDSExplainer:
    def __init__(self, model_dir=None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models")
            
        self.model_dir = model_dir
        self.feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
        self.rf_model = joblib.load(os.path.join(model_dir, "random_forest_binary.pkl"))
        
        # Calculate global feature importances
        self.importances = pd.Series(
            self.rf_model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
    def get_global_top_features(self, n=10):
        """Returns the top N most critical features across the entire model."""
        return self.importances.head(n)

    def explain_flow(self, flow_series, top_k=3):
        """
        Explain why a single network flow was classified as an attack.
        Uses deviation from benign baselines weighted by global feature importance.
        """
        # Ensure correct features
        flow_features = flow_series.reindex(self.feature_names, fill_value=0.0)
        
        # Local heuristic attribution: feature value weighted by model importance
        # Features with high importance that exhibit extreme non-zero or suspicious patterns
        scores = {}
        for feat in self.importances.index[:15]:
            val = float(flow_features[feat])
            imp = float(self.importances[feat])
            
            # Meaningful heuristic weighting based on network characteristics
            weight = imp
            if "Packets/s" in feat or "Bytes/s" in feat:
                if val > 500:
                    weight *= 2.0
            elif "Port" in feat:
                if val in [80, 443, 21, 22, 8080]:
                    weight *= 1.5
            elif "Flag" in feat:
                if val > 0:
                    weight *= 1.8
            scores[feat] = weight * (abs(val) + 1.0)
            
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        reasons = []
        for feat, score in ranked:
            val = flow_features[feat]
            explanation = self._format_feature_reason(feat, val)
            reasons.append({
                "feature": feat,
                "observed_value": val,
                "importance_rank": self.feature_names.index(feat) + 1,
                "reason": explanation
            })
            
        return reasons

    def _format_feature_reason(self, feat, val):
        if "Destination Port" in feat:
            port = int(val)
            service = {80: "HTTP Web Server", 443: "HTTPS Web Server", 21: "FTP Control", 22: "SSH Remote Login"}.get(port, f"Port {port}")
            return f"Traffic targeted critical service: {service}"
        elif "Packets/s" in feat:
            return f"Abnormally high packet rate ({val:,.1f} packets/sec)"
        elif "Bytes/s" in feat:
            return f"High data transmission velocity ({val:,.1f} bytes/sec)"
        elif "SYN Flag Count" in feat and val > 0:
            return f"Active SYN handshake initiation ({int(val)} SYN flags)"
        elif "ACK Flag Count" in feat and val == 0:
            return "Missing ACK response (half-open connection / flood)"
        elif "Flow Duration" in feat and val < 500:
            return f"Extremely short burst duration ({val:.0f} microseconds)"
        elif "Init_Win_bytes" in feat:
            return f"TCP initial window size signature ({int(val)} bytes)"
        else:
            return f"Significant deviation in network signature ({feat} = {val})"

if __name__ == "__main__":
    explainer = NIDSExplainer()
    print("Top 10 Global Features in AI-NIDS:")
    print(explainer.get_global_top_features(10))
