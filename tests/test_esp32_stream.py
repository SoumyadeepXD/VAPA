"""
VAPA Hardware Diagnostic: ESP32 UART Telemetry Stream (/dev/ttyTHS1)
Tests real-time 100 Hz ingestion of 4x FSR forces, MyoWare 2.0 EMG, and EEG.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosignals.esp32_serial_receiver import AsyncESP32Receiver


def main():
    force_mock = "--mock" in sys.argv
    port = "/dev/ttyTHS1" if not force_mock else "MOCK"
    baud = 115200

    print("=" * 75)
    print(" VAPA DIAGNOSTIC: ESP32 Bio-Signal & Tactile Telemetry Receiver")
    print("=" * 75)
    print(f"Connecting to ESP32 on {port} @ {baud} Baud (force_mock={force_mock})...\n")

    receiver = AsyncESP32Receiver(port=port, baud_rate=baud, force_mock=force_mock)
    receiver.start()

    print("Listening for incoming 100 Hz JSON frames (Press Ctrl+C to stop)...\n")

    try:
        last_print = time.time()
        while True:
            now = time.time()
            if now - last_print >= 0.1:  # 10 Hz display rate
                last_print = now
                frame = receiver.get_latest_frame()
                stats = receiver.get_stats()

                f = frame.fsr_forces_n
                sys.stdout.write(
                    f"\r[Seq:{frame.seq:6d}] "
                    f"FSR(N): [Th:{f[0]:4.1f} In:{f[1]:4.1f} Mi:{f[2]:4.1f} Ri:{f[3]:4.1f}] "
                    f"Tot:{frame.total_grip_force_n:4.1f}N | "
                    f"EMG_Act:{frame.emg_activation:4.2f} (V:{frame.emg_volts:.2f}V) | "
                    f"EEG_V:{frame.eeg_volts:.2f}V | "
                    f"Rx:{stats['packets_received']} Drop:{stats['packets_dropped']}"
                )
                sys.stdout.flush()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
    finally:
        receiver.stop()
        print("Receiver stopped.")


if __name__ == "__main__":
    main()
