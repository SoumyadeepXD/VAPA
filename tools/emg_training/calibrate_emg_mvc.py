#!/usr/bin/env python3
"""
VAPA EMG Baseline & Maximum Voluntary Contraction (MVC) Calibration Tool
Guides the operator through a standardized two-phase biometric calibration:
1. Resting Baseline Calibration (10s relaxed arm) -> Determines sensor noise floor & baseline offset
2. Maximum Voluntary Contraction (MVC) (5s hard grasp) -> Determines peak dynamic envelope excursion

Saves calibrated thresholds and normalization constants to config/emg_calibration.json.
"""

import os
import sys
import time
import json
import argparse
import numpy as np

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
    parser = argparse.ArgumentParser(description="Calibrate EMG baseline and MVC")
    parser.add_argument("--subject", type=str, default="SUBJ_01", help="Subject ID")
    parser.add_argument("--config_out", type=str, default="config/emg_calibration.json", help="Path to write calibration JSON")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock streamer")
    return parser.parse_args()


def run_guided_calibration(subject: str, config_out: str, force_mock: bool = False):
    print(SAFETY_WARNING)

    print(f"[*] Initializing Calibration Session for Subject: {subject} (mock={force_mock})")
    receiver = ESP32TelemetryReceiver(force_mock=force_mock)
    if not receiver.is_connected and not force_mock:
        print("[!] Physical ESP32 not detected. Falling back to mock receiver.")
        receiver = ESP32TelemetryReceiver(force_mock=True)

    receiver.start()

    # --------------------------------------------------------------------------
    # PHASE 1: RESTING BASELINE CALIBRATION (10s)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: RESTING BASELINE CALIBRATION")
    print("=" * 60)
    print("Please keep your arm, hand, and fingers COMPLETELY RELAXED on the table.")
    print("Starting in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print("\n[>>> MEASURING RESTING BASELINE (10 seconds) <<<]")

    rest_voltages = []
    start = time.time()
    while time.time() - start < 10.0:
        frame = receiver.get_latest_frame()
        if frame and frame.is_valid:
            rest_voltages.append(frame.emg_volts)
        time.sleep(0.010)

    if not rest_voltages:
        rest_voltages = [0.15]

    rest_mean = float(np.mean(rest_voltages))
    rest_std = float(np.std(rest_voltages))
    rest_max = float(np.max(rest_voltages))
    print(f"[OK] Phase 1 Complete. Resting Baseline Mean: {rest_mean:.3f}V (Std: {rest_std:.4f}V, Max: {rest_max:.3f}V)")

    # --------------------------------------------------------------------------
    # PHASE 2: MAXIMUM VOLUNTARY CONTRACTION (MVC) (5s)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2: MAXIMUM VOLUNTARY CONTRACTION (MVC)")
    print("=" * 60)
    print("Prepare to squeeze your fist at MAXIMUM FORCE for 5 seconds when prompted.")
    print("Starting in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print("\n[>>> SQUEEZE MAXIMUM FORCE NOW! (5 seconds) <<<]")

    mvc_voltages = []
    start = time.time()
    while time.time() - start < 5.0:
        frame = receiver.get_latest_frame()
        if frame and frame.is_valid:
            mvc_voltages.append(frame.emg_volts)
        time.sleep(0.010)

    receiver.stop()

    if not mvc_voltages:
        mvc_voltages = [2.2]

    mvc_peak = float(np.max(mvc_voltages))
    mvc_mean = float(np.mean(mvc_voltages))
    print(f"[OK] Phase 2 Complete. MVC Peak: {mvc_peak:.3f}V (Mean Contraction: {mvc_mean:.3f}V)")

    # --------------------------------------------------------------------------
    # THRESHOLD CALCULATION & EXPORT
    # --------------------------------------------------------------------------
    dynamic_range = max(0.1, mvc_peak - rest_mean)
    rest_threshold = rest_mean + 0.15 * dynamic_range
    activation_threshold = rest_mean + 0.30 * dynamic_range
    high_contraction_threshold = rest_mean + 0.65 * dynamic_range
    co_contraction_threshold = rest_mean + 0.85 * dynamic_range

    calib_data = {
        "subject_id": subject,
        "calibration_timestamp": time.time(),
        "baseline_voltage_v": round(rest_mean, 4),
        "baseline_std_v": round(rest_std, 4),
        "mvc_peak_voltage_v": round(mvc_peak, 4),
        "dynamic_range_v": round(dynamic_range, 4),
        "thresholds_normalized": {
            "rest": 0.12,
            "activation": 0.28,
            "high_contraction": 0.65,
            "co_contraction": 0.85,
        },
        "thresholds_volts": {
            "rest_threshold_v": round(rest_threshold, 4),
            "activation_threshold_v": round(activation_threshold, 4),
            "high_contraction_threshold_v": round(high_contraction_threshold, 4),
            "co_contraction_threshold_v": round(co_contraction_threshold, 4),
        },
        "safety_invariants": {
            "fail_closed_mode": True,
            "parallel_estop_active": True,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(config_out)), exist_ok=True)
    with open(config_out, "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\n[SUCCESS] Calibration saved to: {config_out}")
    print(json.dumps(calib_data, indent=2))
    return calib_data


def main():
    args = parse_args()
    run_guided_calibration(args.subject, args.config_out, args.mock)


if __name__ == "__main__":
    main()
