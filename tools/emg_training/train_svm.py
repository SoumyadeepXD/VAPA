import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load dataset
df = pd.read_csv(
    "data/features_frequency.csv"
)


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
# Temporal split
# --------------------------------------------------

train_parts = []
test_parts = []


for label in df["Label"].unique():

    gesture_data = df[
        df["Label"] == label
    ].reset_index(drop=True)

    split_index = int(
        len(gesture_data) * 0.70
    )

    train_parts.append(
        gesture_data.iloc[:split_index]
    )

    test_parts.append(
        gesture_data.iloc[split_index:]
    )


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)


X_train = train_df[feature_columns]
y_train = train_df["Label"]

X_test = test_df[feature_columns]
y_test = test_df["Label"]


# --------------------------------------------------
# Standardization
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# --------------------------------------------------
# SVM
# --------------------------------------------------

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(
    X_test
)


# --------------------------------------------------
# Results
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("======================================")
print("SVM EMG CLASSIFICATION")
print("======================================")

print()

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print()

print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print()

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)