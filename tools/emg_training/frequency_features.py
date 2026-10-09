import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
OUTPUT_FILE = "data/features_frequency.csv"

FS = 500

WINDOW_SIZE = 500
STEP_SIZE = 250


def extract_features(signal):
    """
    Extract time-domain + frequency-domain EMG features.
    """

    # Remove DC component
    signal = signal - np.mean(signal)

    # -----------------------------
    # TIME-DOMAIN FEATURES
    # -----------------------------

    rms = np.sqrt(np.mean(signal ** 2))

    mav = np.mean(np.abs(signal))

    variance = np.var(signal)

    std = np.std(signal)

    waveform_length = np.sum(np.abs(np.diff(signal)))

    threshold = 0.01 * std

    zero_crossings = np.sum(
        ((signal[:-1] * signal[1:]) < 0) &
        (np.abs(signal[:-1] - signal[1:]) >= threshold)
    )

    # -----------------------------
    # FREQUENCY-DOMAIN FEATURES
    # -----------------------------

    # FFT
    fft_values = np.fft.rfft(signal)

    # Magnitude
    magnitude = np.abs(fft_values)

    # Frequencies
    frequencies = np.fft.rfftfreq(len(signal), d=1 / FS)

    # Ignore DC component
    magnitude[0] = 0

    # Power spectrum
    power = magnitude ** 2

    total_power = np.sum(power)

    if total_power > 0:

        # Mean Frequency
        mean_frequency = np.sum(
            frequencies * power
        ) / total_power

        # Median Frequency
        cumulative_power = np.cumsum(power)

        median_frequency = frequencies[
            np.searchsorted(
                cumulative_power,
                total_power / 2
            )
        ]

    else:

        mean_frequency = 0
        median_frequency = 0

    # Total signal energy
    energy = np.sum(signal ** 2)

    # Peak-to-peak amplitude
    peak_to_peak = np.max(signal) - np.min(signal)

    return [
        rms,
        mav,
        variance,
        std,
        waveform_length,
        zero_crossings,
        mean_frequency,
        median_frequency,
        energy,
        peak_to_peak
    ]


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

    signal = pd.to_numeric(
        df["emg"],
        errors="coerce"
    ).dropna().values

    # Remove first 5 seconds from gesture recordings
    if label != "REST":

        skip_samples = 5 * FS

        signal = signal[skip_samples:]


    # Sliding windows
    for start in range(
        0,
        len(signal) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        window = signal[
            start:start + WINDOW_SIZE
        ]

        features = extract_features(window)

        all_features.append(
            features + [label]
        )


columns = [
    "RMS",
    "MAV",
    "Variance",
    "STD",
    "Waveform_Length",
    "Zero_Crossings",
    "Mean_Frequency",
    "Median_Frequency",
    "Energy",
    "Peak_to_Peak",
    "Label"
]


features_df = pd.DataFrame(
    all_features,
    columns=columns
)


features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("===================================")
print("FREQUENCY FEATURE EXTRACTION")
print("===================================")

print()

print("Saved to:", OUTPUT_FILE)

print()

print(
    "Dataset shape:",
    features_df.shape
)

print()

print("Samples per gesture:")

print(
    features_df["Label"].value_counts()
)

print()

print("First few rows:")

print(
    features_df.head()
)