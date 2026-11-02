import pandas as pd
import os
import time
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# SETTINGS
# =========================================================

DATA_DIR = "dataset/processed/prepared"
MODEL_DIR = "models"

RANDOM_STATE = 42

# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("RANDOM FOREST - BINARY NIDS")
print("=" * 70)

print("\nLoading training data...")

X_train = pd.read_csv(
    os.path.join(DATA_DIR, "X_train.csv")
)

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
).squeeze()

print("Training data:", X_train.shape)
print("Training labels:", y_train.shape)

print("\nLoading validation data...")

X_validation = pd.read_csv(
    os.path.join(DATA_DIR, "X_validation.csv")
)

y_validation = pd.read_csv(
    os.path.join(DATA_DIR, "y_validation.csv")
).squeeze()

print("Validation data:", X_validation.shape)
print("Validation labels:", y_validation.shape)

# =========================================================
# CREATE MODEL
# =========================================================

print("\n" + "=" * 70)
print("CREATING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Number of trees:", model.n_estimators)
print("Features used:", X_train.shape[1])
print("Parallel CPU workers: all available")

# =========================================================
# TRAIN
# =========================================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

start_time = time.time()

model.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print("\nTraining completed!")
print(f"Training time: {training_time:.2f} seconds")

# =========================================================
# VALIDATION PREDICTION
# =========================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

start_time = time.time()

y_pred = model.predict(X_validation)

prediction_time = time.time() - start_time

print(f"Prediction time: {prediction_time:.2f} seconds")

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_validation,
    y_pred
)

precision = precision_score(
    y_validation,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_validation,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    y_pred,
    zero_division=0
)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_validation,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_validation,
    y_pred
)

print("\n              Predicted")
print("              BENIGN  ATTACK")
print(f"Actual BENIGN  {cm[0][0]:7d}  {cm[0][1]:7d}")
print(f"Actual ATTACK  {cm[1][0]:7d}  {cm[1][1]:7d}")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("\n" + "=" * 70)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 70)

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    importance.head(20).to_string(index=False)
)

# =========================================================
# SAVE MODEL
# =========================================================

model_path = os.path.join(
    MODEL_DIR,
    "random_forest_binary.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("Model:", model_path)

print("\n" + "=" * 70)
print("RANDOM FOREST TRAINING COMPLETE")
print("=" * 70)