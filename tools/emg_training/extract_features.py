import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
OUTPUT_FILE = "data/features.csv"

# Your timestamps showed approximately 2000 microseconds between samples
# = 500 Hz sampling rate.
FS = 500

# 1-second windows
WINDOW_SIZE = 500

# 50% overlap
STEP_SIZE = 250


def extract_features(signal):
    """
    Extract time-domain EMG features from one window.
    """

    # Remove DC/baseline component
    signal = signal - np.mean(signal)

    # RMS
    rms = np.sqrt(np.mean(signal ** 2))

    # Mean Absolute Value
    mav = np.mean(np.abs(signal))

    # Variance
    variance = np.var(signal)

    # Standard deviation
    std = np.std(signal)

    # Waveform Length
    waveform_length = np.sum(np.abs(np.diff(signal)))

    # Zero Crossings
    threshold = 0.01 * std

    zero_crossings = np.sum(
        ((signal[:-1] * signal[1:]) < 0) &
        (np.abs(signal[:-1] - signal[1:]) >= threshold)
    )

    return [
        rms,
        mav,
        variance,
        std,
        waveform_length,
        zero_crossings
    ]


# Files and labels
files = {
    "rest.csv": "REST",
    "fist.csv": "FIST",
    "open.csv": "OPEN",
    "wrist_flexion.csv": "WRIST_FLEXION",
    "wrist_extension.csv": "WRIST_EXTENSION"
}

all_features = []


for filename, label in files.items():

    filepath = os.path.join(DATA_DIR, filename)

    print(f"Processing {filename}...")

    df = pd.read_csv(filepath)

    signal = pd.to_numeric(df["emg"], errors="coerce").dropna().values

    # Remove first 5 seconds from gesture recordings.
    # Those 5 seconds were deliberately recorded with the hand relaxed.
    if label != "REST":
        skip_samples = 5 * FS
        signal = signal[skip_samples:]

    # Extract overlapping windows
    for start in range(0, len(signal) - WINDOW_SIZE + 1, STEP_SIZE):

        window = signal[start:start + WINDOW_SIZE]

        features = extract_features(window)

        all_features.append(
            features + [label]
        )


# Create dataframe
columns = [
    "RMS",
    "MAV",
    "Variance",
    "STD",
    "Waveform_Length",
    "Zero_Crossings",
    "Label"
]

features_df = pd.DataFrame(all_features, columns=columns)

# Save
features_df.to_csv(OUTPUT_FILE, index=False)

print()
print("Feature extraction complete!")
print("Saved to:", OUTPUT_FILE)
print()
print("Dataset shape:", features_df.shape)
print()
print("Samples per gesture:")
print(features_df["Label"].value_counts())
print()
print(features_df.head())