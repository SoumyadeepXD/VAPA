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
    dur = None
    if "--duration" in sys.argv:
        try:
            dur = float(sys.argv[sys.argv.index("--duration") + 1])
        except (IndexError, ValueError):
            dur = 2.0

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
        start_t = time.time()
        last_print = time.time()
        while True:
            if dur is not None and (time.time() - start_t) >= dur:
                print(f"\nReached target duration {dur}s. Exiting cleanly.")
                break
            now = time.time()
            if now - last_print >= 0.1:  # 10 Hz display rate
                last_print = now
                frame = receiver.get_latest_frame()
                stats = receiver.get_stats()

                f = frame.fsr_forces_n
                pi_f = f[4] if len(f) > 4 else 0.0
                enc_deg = frame.encoder_angle_deg
                sys.stdout.write(
                    f"\r[Seq:{frame.seq:6d}] "
                    f"FSR(N): [Th:{f[0]:4.1f} In:{f[1]:4.1f} Mi:{f[2]:4.1f} Ri:{f[3]:4.1f} Pi:{pi_f:4.1f}] "
                    f"Tot:{frame.total_grip_force_n:4.1f}N | "
                    f"Enc:{enc_deg:5.1f}° | "
                    f"Flex:{frame.emg_flex_activation:4.2f} Ext:{frame.emg_ext_activation:4.2f} | "
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
