import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load feature dataset
df = pd.read_csv("data/features.csv")

feature_columns = [
    "RMS",
    "MAV",
    "Variance",
    "STD",
    "Waveform_Length",
    "Zero_Crossings"
]

X = df[feature_columns]
y = df["Label"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Prediction
y_pred = model.predict(X_test_scaled)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print()
print("===================================")
print("EMG GESTURE CLASSIFICATION RESULTS")
print("===================================")

print()
print("Accuracy:", round(accuracy * 100, 2), "%")

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))


print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print()
print("Feature Importance:")

for feature, importance in sorted(
    zip(feature_columns, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature}: {importance:.4f}")