import os
import sys
import pandas as pd

# Add repo root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from detection.unified_detector import UnifiedNIDSDetector

def test_inference():
    print("=" * 70)
    print("TESTING UNIFIED AI-NIDS DETECTOR ON REAL NETWORK FLOWS")
    print("=" * 70)
    
    detector = UnifiedNIDSDetector()
    live_csv = os.path.join(ROOT, "dataset", "processed", "live_flows.csv")
    
    if not os.path.exists(live_csv):
        print(f"File not found: {live_csv}")
        return
        
    flows = pd.read_csv(live_csv)
    print(f"\nLoaded {len(flows)} extracted flows from {live_csv}.")
    
    results = detector.predict(flows)
    
    print("\n--- INFERENCE RESULTS ---")
    print(results[[
        "flow_index", "verdict", "severity", "attack_probability",
        "predicted_category", "category_confidence", "zero_day_anomaly"
    ]].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_inference()
