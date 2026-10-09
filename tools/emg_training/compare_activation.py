import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
rest = pd.read_csv("data/rest.csv")
fist = pd.read_csv("data/fist.csv")

# Remove DC/baseline component
rest_signal = rest["emg"].values - rest["emg"].mean()
fist_signal = fist["emg"].values - fist["emg"].mean()

# Calculate RMS using 250-sample windows
window = 250

rest_rms = np.sqrt(
    pd.Series(rest_signal ** 2).rolling(window).mean()
)

fist_rms = np.sqrt(
    pd.Series(fist_signal ** 2).rolling(window).mean()
)

# Plot
plt.figure(figsize=(12, 5))

plt.plot(rest_rms, label="REST")
plt.plot(fist_rms, label="FIST")

plt.title("EMG Activation: REST vs FIST")
plt.xlabel("Sample Window")
plt.ylabel("RMS Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print average RMS
print("REST average RMS:", round(rest_rms.mean(), 2))
print("FIST average RMS:", round(fist_rms.mean(), 2))