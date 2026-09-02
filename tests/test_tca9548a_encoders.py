"""
VAPA Hardware Diagnostic: TCA9548A I2C Multiplexer & 4x AS5600 Encoders
Polls all 4 magnetic rotary encoders on /dev/i2c-1 (Address 0x70 -> 0x36)
and displays raw counts, calibrated angles, and magnet field status.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivers.tca9548a_as5600 import AS5600EncoderMux, ENCODER_CHANNEL_MAP


def main():
    force_mock = "--mock" in sys.argv
    print("=" * 75)
    print(" VAPA DIAGNOSTIC: TCA9548A Multiplexer & AS5600 Magnetic Encoders")
    print("=" * 75)
    print(f"Initializing I2C MUX on /dev/i2c-1 (force_mock={force_mock})...\n")

    mux = AS5600EncoderMux(bus_num=1, force_mock=force_mock)

    print("Channel Mapping:")
    for ch, cfg in ENCODER_CHANNEL_MAP.items():
        print(f"  MUX Ch {ch}: {cfg['name']} ({cfg['description']})")

    print("\nReading live encoder streams at 20 Hz (Press Ctrl+C to stop)...\n")

    try:
        while True:
            t0 = time.time()
            readings = mux.read_all_encoders()

            status_line = ""
            for ch, cfg in ENCODER_CHANNEL_MAP.items():
                name = cfg["name"]
                deg = readings[name]
                raw = mux.last_raw_counts.get(ch, 0)
                status = mux.check_magnet_status(ch)
                mag_ok = "[MAG OK]" if status["magnet_detected"] else "[NO MAG]"
                status_line += f"{name[:10]}: {deg:5.1f}° (Raw:{raw:4d} {mag_ok}) | "

            sys.stdout.write(f"\r{status_line}")
            sys.stdout.flush()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
    finally:
        mux.close()
        print("TCA9548A connection closed.")


if __name__ == "__main__":
    main()
