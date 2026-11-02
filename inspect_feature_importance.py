
import joblib
import pandas as pd

model = joblib.load("models/random_forest_binary_v3.pkl")

importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 20 IMPORTANT FEATURES")
print(importance.head(20).to_string(index=False))

importance.to_csv(
    "models/feature_importance_report.csv",
    index=False
)

print("\nReport saved to models/feature_importance_report.csv")
