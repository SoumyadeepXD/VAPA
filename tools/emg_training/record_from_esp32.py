#!/usr/bin/env python3
"""
VAPA ESP32 EMG Stream Recorder
Acquires labeled surface electromyography (sEMG) datasets directly through
the live VAPA deployed hardware path (ESP32 UART / ADS1115 A0 / MyoWare 2.0).

Supports both physical UART hardware (/dev/ttyTHS1, /dev/ttyUSB0) and mock synthetic streaming.
Outputs CSV datasets matching data/emg/raw format with complete biomedical provenance metadata.
"""

import os
import sys
import time
import argparse
import datetime
from typing import Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from biosignals.esp32_serial_receiver import ESP32TelemetryReceiver

SAFETY_WARNING = """
================================================================================
           CRITICAL BIOMEDICAL ELECTRICAL SAFETY WARNING
================================================================================
1. RUN ON BATTERY POWER ONLY WHEN ELECTRODES ARE ATTACHED TO HUMAN SKIN.
2. DO NOT CONNECT JETSON, ESP32, OR TEST RIG TO WALL-POWERED EQUIPMENT OR
   A MAINS-POWERED LAPTOP / PC (ISOLATION FAULT HAZARD).
3. POTENTIAL GROUND LOOPS THROUGH ELECTRODES PRESENT AN EXTREME ELECTRIC SHOCK
   AND VENTRICULAR FIBRILLATION HAZARD.
4. KEEP THE PHYSICAL HARDWARE EMERGENCY STOP (OR KEYBOARD 'E' KEY) IMMEDIATELY
   WITHIN REACH AT ALL TIMES DURING RECORDING.
================================================================================
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Record labeled EMG from ESP32 telemetry path")
    parser.add_argument("--gesture", type=str, default="REST", choices=["REST", "FIST", "OPEN", "WRIST_FLEXION", "WRIST_EXTENSION", "PINCH", "CUSTOM"], help="Gesture label")
    parser.add_argument("--subject", type=str, default="SUBJ_01", help="Anonymized Subject ID (never use real names)")
    parser.add_argument("--session", type=int, default=1, help="Session index")
    parser.add_argument("--placement", type=str, default="flexor_digitorum_superficialis", help="Anatomical electrode location")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds")
    parser.add_argument("--output", type=str, default="", help="Custom output CSV path")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock streamer (no hardware needed)")
    return parser.parse_args()


def record_emg_session(
    gesture: str,
    subject: str,
    session: int,
    placement: str,
    duration_s: float,
    output_path: str,
    force_mock: bool = False,
):
    print(SAFETY_WARNING)

    if not output_path:
        out_dir = "data/emg/raw"
        os.makedirs(out_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(out_dir, f"{gesture.lower()}_{subject.lower()}_s{session}_{timestamp_str}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[*] Initializing ESP32 Telemetry Receiver (mock={force_mock})...")
    receiver = ESP32TelemetryReceiver(force_mock=force_mock)
    if not receiver.is_connected and not force_mock:
        print("[!] Physical ESP32 not detected on UART candidates. Falling back to mock mode.")
        receiver = ESP32TelemetryReceiver(force_mock=True)

    receiver.start()

    print(f"[*] Preparing to record gesture '{gesture}' for {duration_s:.1f}s.")
    print("[*] 3-second countdown starting. Relax arm...")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        time.sleep(1.0)
    print(f"\n[>>> START RECORDING: PERFORM '{gesture}' NOW <<<]\n")

    recorded_rows = []
    start_time = time.time()
    t_start_us = int(start_time * 1e6)

    try:
        while time.time() - start_time < duration_s:
            frame = receiver.get_latest_frame()
            if frame and frame.is_valid:
                now_us = int(time.time() * 1e6)
                # Convert envelope volts (0-3.3V) to 12-bit ADC count equivalent (0-4095) for consistency
                raw_adc_equivalent = int((frame.emg_volts / 3.3) * 4095)
                recorded_rows.append((now_us, raw_adc_equivalent, frame.emg_volts, frame.emg_activation))
            time.sleep(0.010)  # 100 Hz polling loop
    finally:
        receiver.stop()

    print(f"\n[*] Recording complete. Captured {len(recorded_rows)} samples.")

    # Write formatted CSV with standard header + metadata comments
    with open(output_path, "w") as f:
        f.write(f"# metadata_subject_id: {subject}\n")
        f.write(f"# metadata_session_id: {session}\n")
        f.write(f"# metadata_gesture: {gesture}\n")
        f.write(f"# metadata_placement: {placement}\n")
        f.write(f"# metadata_sampling_rate_hz: 100.0\n")
        f.write(f"# metadata_date: {datetime.date.today().isoformat()}\n")
        f.write(f"# metadata_mock_stream: {force_mock}\n")
        f.write("timestamp_us,emg\n")
        for r in recorded_rows:
            f.write(f"{r[0]},{r[1]}\n")

    print(f"[OK] Saved recorded dataset to: {output_path}")
    return output_path


def main():
    args = parse_args()
    record_emg_session(
        gesture=args.gesture,
        subject=args.subject,
        session=args.session,
        placement=args.placement,
        duration_s=args.duration,
        output_path=args.output,
        force_mock=args.mock,
    )


if __name__ == "__main__":
    main()
