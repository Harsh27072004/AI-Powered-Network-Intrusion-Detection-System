import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

X = pd.read_csv("dataset/processed/prepared_v3/X_validation.csv")
y = pd.read_csv("dataset/processed/prepared_v3/y_validation.csv").iloc[:, 0]

model = joblib.load("models/random_forest_binary_v3.pkl")

attack_index = list(model.classes_).index(1)
probabilities = model.predict_proba(X)[:, attack_index]

print("ROC-AUC:", round(roc_auc_score(y, probabilities), 4))

print("\nTHRESHOLD | PRECISION | RECALL | F1 | SPECIFICITY | FPR")

for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)

    print(
        f"{threshold:.2f} | {precision:.4f} | "
        f"{recall:.4f} | {f1:.4f} | "
        f"{specificity:.4f} | {fpr:.4f}"
    )