import serial
import csv
import time

PORT = "COM3"
BAUD_RATE = 115200
DURATION = 30

OUTPUT_FILE = "data/wrist_extension.csv"

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

# Give ESP32 time to initialize
time.sleep(2)

# Clear old/boot data from the serial buffer
ser.reset_input_buffer()

print("Starting EMG recording...")
print("Keep your right hand completely relaxed for the first 5 seconds.")

start_time = time.time()

with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp_us", "emg"])

    while time.time() - start_time < DURATION:

        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        parts = line.split(",")

        # Accept ONLY: numeric timestamp + numeric EMG
        if len(parts) != 2:
            continue

        try:
            timestamp = int(parts[0])
            emg = int(parts[1])
        except ValueError:
            continue

        writer.writerow([timestamp, emg])

ser.close()

print("Recording complete!")
print("Saved to:", OUTPUT_FILE)