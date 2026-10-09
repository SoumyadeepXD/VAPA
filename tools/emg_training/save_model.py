import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# Load feature dataset
# --------------------------------------------------

df = pd.read_csv("data/features_frequency.csv")


feature_columns = [
    "RMS",
    "MAV",
    "Variance",
    "STD",
    "Waveform_Length",
    "Zero_Crossings",
    "Mean_Frequency",
    "Median_Frequency",
    "Energy",
    "Peak_to_Peak"
]


# --------------------------------------------------
# Use the complete current dataset
# --------------------------------------------------

X = df[feature_columns]
y = df["Label"]


# --------------------------------------------------
# Standardize
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# Train the best current model
# --------------------------------------------------

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True
)

model.fit(X_scaled, y)


# --------------------------------------------------
# Save model + scaler + feature list
# --------------------------------------------------

joblib.dump(
    model,
    "svm_emg_model.pkl"
)

joblib.dump(
    scaler,
    "emg_scaler.pkl"
)

joblib.dump(
    feature_columns,
    "feature_columns.pkl"
)


print()
print("======================================")
print("EMG MODEL SAVED")
print("======================================")

print()
print("Model  : svm_emg_model.pkl")
print("Scaler : emg_scaler.pkl")
print("Features: feature_columns.pkl")

print()
print("Classes:")
print(model.classes_)