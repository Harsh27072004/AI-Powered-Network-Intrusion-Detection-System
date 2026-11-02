import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from detection.unified_detector import UnifiedNIDSDetector

def test_attack_families():
    print("=" * 70)
    print("TESTING UNIFIED DETECTOR ACROSS UNSEEN ATTACK FAMILIES")
    print("=" * 70)
    
    detector = UnifiedNIDSDetector()
    data_dir = os.path.join(ROOT, "dataset", "processed", "prepared")
    
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test_mul = pd.read_csv(os.path.join(data_dir, "y_test_multiclass.csv")).squeeze()
    
    # Pick 2 samples from each attack category
    samples = []
    ground_truths = []
    
    categories = sorted(y_test_mul.unique())
    for cat in categories:
        indices = y_test_mul[y_test_mul == cat].index[:2]
        for idx in indices:
            samples.append(X_test.iloc[idx])
            ground_truths.append(cat)
            
    df_samples = pd.DataFrame(samples).reset_index(drop=True)
    
    results = detector.predict(df_samples)
    results["ground_truth"] = ground_truths
    
    print("\n--- ATTACK IDENTIFICATION BENCHMARK ---")
    cols = ["ground_truth", "verdict", "severity", "attack_probability", "predicted_category", "category_confidence"]
    print(results[cols].to_string(index=False))
    
    match_count = (results["ground_truth"] == results["predicted_category"]).sum()
    print(f"\nExact Category Match: {match_count} / {len(results)} ({match_count/len(results)*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    test_attack_families()
