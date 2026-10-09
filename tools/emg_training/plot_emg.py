import pandas as pd
import matplotlib.pyplot as plt

# Load REST EMG data
data = pd.read_csv("data/fist.csv")

# Convert timestamp from microseconds to seconds
time = (data["timestamp_us"] - data["timestamp_us"].iloc[0]) / 1_000_000

# Plot EMG
plt.figure(figsize=(12, 5))
plt.plot(time, data["emg"], linewidth=0.8)

plt.title("EMG Signal - FIST")
plt.xlabel("Time (seconds)")
plt.ylabel("EMG ADC Value")
plt.grid(True)

plt.tight_layout()
plt.show()