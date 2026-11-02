import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ==============================================================================
# HYBRID MODEL TRAINER: BINARY, MULTI-CLASS & ZERO-DAY ANOMALY DETECTOR
# AI-NIDS Professional Pipeline
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed", "prepared")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 75)
print("PHASE 2: TRAINING HYBRID AI-NIDS ENGINES (LEAK-FREE)")
print("=" * 75)

# ------------------------------------------------------------------------------
# 1. LOAD PREPARED LEAK-FREE DATA
# ------------------------------------------------------------------------------
print("\n[1/4] Loading training and test datasets...")
start_load = time.time()

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train_bin = pd.read_csv(os.path.join(DATA_DIR, "y_train_binary.csv")).squeeze()
y_test_bin = pd.read_csv(os.path.join(DATA_DIR, "y_test_binary.csv")).squeeze()

y_train_mul = pd.read_csv(os.path.join(DATA_DIR, "y_train_multiclass.csv")).squeeze()
y_test_mul = pd.read_csv(os.path.join(DATA_DIR, "y_test_multiclass.csv")).squeeze()

print(f"Loaded X_train: {X_train.shape} | X_test: {X_test.shape} in {time.time()-start_load:.1f}s")
feature_names = list(X_train.columns)

# Save feature list for inference verification
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

# ------------------------------------------------------------------------------
# 2. MODEL A: BINARY INTRUSION CLASSIFIER (Random Forest)
# ------------------------------------------------------------------------------
print("\n" + "=" * 75)
print("[2/4] Training Model A: Binary Classifier (Random Forest)")
print("=" * 75)

rf_bin = RandomForestClassifier(
    n_estimators=80,
    max_depth=22,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

start_rf = time.time()
rf_bin.fit(X_train, y_train_bin)
rf_duration = time.time() - start_rf
print(f"Binary Random Forest training complete in {rf_duration:.1f} seconds.")

# Evaluate Binary Model on Unseen Test Set
y_pred_bin = rf_bin.predict(X_test)
acc_bin = accuracy_score(y_test_bin, y_pred_bin)
f1_bin = f1_score(y_test_bin, y_pred_bin)

print("\n--- Model A Evaluation (Unseen Test Set - Zero Leakage) ---")
print(f"Accuracy:  {acc_bin * 100:.3f}%")
print(f"F1-Score:  {f1_bin * 100:.3f}%")
print("\nClassification Report (Binary):")
print(classification_report(y_test_bin, y_pred_bin, target_names=["BENIGN (0)", "ATTACK (1)"], digits=4))

cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
print("Confusion Matrix:")
print(f"[[TN={cm_bin[0,0]:,}, FP={cm_bin[0,1]:,}],")
print(f" [FN={cm_bin[1,0]:,}, TP={cm_bin[1,1]:,}]]")
far = (cm_bin[0,1] / (cm_bin[0,0] + cm_bin[0,1])) * 100
print(f"False Alarm Rate (FAR): {far:.3f}%")

joblib.dump(rf_bin, os.path.join(MODEL_DIR, "random_forest_binary.pkl"))
print("Saved Model A -> models/random_forest_binary.pkl")

# ------------------------------------------------------------------------------
# 3. MODEL B: MULTI-CLASS ATTACK CLASSIFIER (HistGradientBoosting)
# ------------------------------------------------------------------------------
print("\n" + "=" * 75)
print("[3/4] Training Model B: Multi-Class Attack Classifier (Gradient Boosting)")
print("=" * 75)

# Label encoding for multi-class
class_names = sorted(y_train_mul.unique())
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for idx, name in enumerate(class_names)}

y_train_mul_enc = y_train_mul.map(class_to_idx)
y_test_mul_enc = y_test_mul.map(class_to_idx)

joblib.dump(idx_to_class, os.path.join(MODEL_DIR, "class_labels.pkl"))

hgb_mul = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=12,
    learning_rate=0.1,
    random_state=42
)

start_hgb = time.time()
hgb_mul.fit(X_train, y_train_mul_enc)
hgb_duration = time.time() - start_hgb
print(f"Multi-Class Gradient Boosting training complete in {hgb_duration:.1f} seconds.")

# Evaluate Multi-Class Model
y_pred_mul_enc = hgb_mul.predict(X_test)
y_pred_mul = pd.Series(y_pred_mul_enc).map(idx_to_class)

acc_mul = accuracy_score(y_test_mul, y_pred_mul)
print("\n--- Model B Evaluation (Multi-Class on Unseen Test Set) ---")
print(f"Multi-Class Accuracy: {acc_mul * 100:.3f}%")
print("\nClassification Report (Multi-Class):")
print(classification_report(y_test_mul, y_pred_mul, target_names=class_names, digits=4))

joblib.dump(hgb_mul, os.path.join(MODEL_DIR, "gradient_boost_multiclass.pkl"))
print("Saved Model B -> models/gradient_boost_multiclass.pkl")

# ------------------------------------------------------------------------------
# 4. MODEL C: ZERO-DAY ANOMALY DETECTOR (Isolation Forest)
# ------------------------------------------------------------------------------
print("\n" + "=" * 75)
print("[4/4] Training Model C: Zero-Day Anomaly Detector (Isolation Forest)")
print("=" * 75)
print("Training exclusively on pure BENIGN baseline network flows...")

# Train strictly on clean benign training traffic
X_train_benign = X_train[y_train_bin == 0]

# Subsample 50,000 benign flows for efficient tree construction
sample_benign = X_train_benign.sample(n=min(50000, len(X_train_benign)), random_state=42)

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

start_iso = time.time()
iso_forest.fit(sample_benign)
iso_duration = time.time() - start_iso
print(f"Isolation Forest training complete in {iso_duration:.1f} seconds.")

# Test on test set (Normal = 1, Anomaly = -1)
# Isolation forest predicts: 1 for inlier (benign), -1 for outlier (attack)
y_pred_iso = iso_forest.predict(X_test)
iso_flagged_attack = (y_pred_iso == -1).astype(int)

# Recall on actual attacks:
attack_recall = ((iso_flagged_attack == 1) & (y_test_bin == 1)).sum() / (y_test_bin == 1).sum()
benign_specificity = ((iso_flagged_attack == 0) & (y_test_bin == 0)).sum() / (y_test_bin == 0).sum()

print("\n--- Model C Zero-Day Anomaly Detection Benchmark ---")
print(f"Attack Detection Rate (Unsupervised): {attack_recall * 100:.2f}%")
print(f"Benign Normalcy Score (Specificity):  {benign_specificity * 100:.2f}%")

joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest_anomaly.pkl"))
print("Saved Model C -> models/isolation_forest_anomaly.pkl")

print("\n" + "=" * 75)
print("ALL HYBRID MODELS TRAINED AND ARTIFACTS PERSISTED SUCCESSFULLY!")
print("=" * 75)
