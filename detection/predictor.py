
import os
import joblib
import pandas as pd

# Resolve the model path relative to the project root,
# regardless of the directory from which this file is imported.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "random_forest_binary.pkl"
)

print("Loading AI-NIDS model...")
model = joblib.load(MODEL_PATH)
print("AI-NIDS model loaded successfully.")


def predict_flow(features):
    """Predict whether one network flow is BENIGN or ATTACK."""

    if isinstance(features, dict):
        features = pd.DataFrame([features])
    elif not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a dictionary or pandas DataFrame")

    if len(features) != 1:
        raise ValueError(
            f"Expected exactly one flow; received {len(features)} rows."
        )

    expected_features = list(model.feature_names_in_)

    missing_features = [
        name for name in expected_features
        if name not in features.columns
    ]
    extra_features = [
        name for name in features.columns
        if name not in expected_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing {len(missing_features)} required features: "
            f"{missing_features}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features: {extra_features}"
        )

    # Match the exact feature order used during training.
    features = features[expected_features].copy()

    # Reject invalid values rather than silently replacing them.
    features = features.apply(pd.to_numeric, errors="raise")

    if features.isna().any().any():
        bad_columns = features.columns[
            features.isna().any()
        ].tolist()
        raise ValueError(f"Missing values found in features: {bad_columns}")

    if not features.map(lambda value: __import__("math").isfinite(value)).all().all():
        raise ValueError("Infinite or non-finite values found in features.")

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if 0 not in classes or 1 not in classes:
        raise ValueError(
            f"Unexpected model classes: {classes}. "
            "Expected 0=BENIGN and 1=ATTACK."
        )

    benign_probability = float(probabilities[classes.index(0)])
    attack_probability = float(probabilities[classes.index(1)])

    if prediction == 1:
        label = "ATTACK"
        confidence = attack_probability
    elif prediction == 0:
        label = "BENIGN"
        confidence = benign_probability
    else:
        raise ValueError(f"Unexpected prediction label: {prediction}")

    return {
        "prediction": label,
        "confidence": confidence,
        "benign_probability": benign_probability,
        "attack_probability": attack_probability,
    }
