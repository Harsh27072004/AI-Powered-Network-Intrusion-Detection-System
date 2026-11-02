
import joblib

features = joblib.load("models/feature_names.pkl")
labels = joblib.load("models/class_labels.pkl")

print("CLASS LABELS:", labels)
print("METADATA FEATURE COUNT:", len(features))

model_names = [
    "random_forest_binary.pkl",
    "random_forest_binary_v2.pkl",
    "source_holdout_tuesday_rf.pkl",
    "source_holdout_friday_ddos_rf.pkl",
]

for name in model_names:
    print("\n" + "=" * 50)
    print("MODEL:", name)

    model = joblib.load("models/" + name)

    print("Feature count:", model.n_features_in_)
    print("Classes:", model.classes_)

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        print("Has feature names: Yes")
        print("Exact feature order matches:", model_features == list(features))
    else:
        print("Has feature names: No")
        print("Exact feature order matches: Cannot verify from metadata")

threshold_file = "models/random_forest_binary_v2_threshold.txt"

print("\nV2 THRESHOLD:")
with open(threshold_file, "r") as file:
    print(file.read())
