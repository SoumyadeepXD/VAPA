import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load feature dataset
df = pd.read_csv("data/features.csv")

# Features used by the model
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


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA → reduce 6 features to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


print("Original features:", X.shape[1])
print("PCA components:", X_pca.shape[1])

print()
print(
    "Variance explained by PC1:",
    round(pca.explained_variance_ratio_[0] * 100, 2),
    "%"
)

print(
    "Variance explained by PC2:",
    round(pca.explained_variance_ratio_[1] * 100, 2),
    "%"
)

print(
    "Total variance explained:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)


# Plot
plt.figure(figsize=(10, 7))

for label in sorted(y.unique()):

    mask = y == label

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=label,
        alpha=0.7
    )


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("EMG Gesture Feature Space - PCA")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()