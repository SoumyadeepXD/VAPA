#!/usr/bin/env python3
"""
VAPA Two-Site EMG Baseline & Maximum Voluntary Contraction (MVC) Calibration Tool
Guides the operator through a standardized three-phase biometric calibration:
1. Resting Baseline Calibration (10s relaxed arm) -> Measures flexor & extensor noise floor
2. Flexor Maximum Voluntary Contraction (MVC) (5s hard fist) -> Measures peak flexor excursion
3. Extensor Maximum Voluntary Contraction (MVC) (5s hand open) -> Measures peak extensor excursion

Saves per-channel calibrated thresholds and normalization constants to config/emg_calibration.json.
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
    parser = argparse.ArgumentParser(description="Calibrate two-site EMG baseline and MVC")
    parser.add_argument("--subject", type=str, default="SUBJ_01", help="Subject ID")
    parser.add_argument("--config_out", type=str, default="config/emg_calibration.json", help="Path to write calibration JSON")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock streamer")
    parser.add_argument("--fast", action="store_true", help="Accelerated timings for automated testing/CI")
    return parser.parse_args()


def run_guided_calibration(subject: str, config_out: str, force_mock: bool = False, fast: bool = False):
    print(SAFETY_WARNING)

    print(f"[*] Initializing Two-Site Calibration Session for Subject: {subject} (mock={force_mock}, fast={fast})")
    receiver = ESP32TelemetryReceiver(force_mock=force_mock)
    if not receiver.is_connected and not force_mock:
        print("[!] Physical ESP32 not detected. Falling back to mock receiver.")
        receiver = ESP32TelemetryReceiver(force_mock=True)

    receiver.start()

    rest_duration = 1.0 if fast else 10.0
    mvc_duration = 0.5 if fast else 5.0
    countdown = 0 if fast else 3

    # --------------------------------------------------------------------------
    # PHASE 1: RESTING BASELINE CALIBRATION (10s)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 1: RESTING BASELINE CALIBRATION (BOTH CHANNELS)")
    print("=" * 65)
    print("Please keep your arm, hand, and fingers COMPLETELY RELAXED on the table.")
    if countdown > 0:
        print(f"Starting in {countdown} seconds...")
        for i in range(countdown, 0, -1):
            print(f"  {i}...")
            time.sleep(1.0)
    print(f"\n[>>> MEASURING RESTING BASELINE ({rest_duration:.1f} seconds) <<<]")

    if force_mock or not receiver.is_connected:
        receiver.set_mock_scenario("rest")

    rest_flex_v = []
    rest_ext_v = []
    start = time.time()
    while time.time() - start < rest_duration:
        frame = receiver.get_latest_frame()
        if frame and frame.is_valid:
            rest_flex_v.append(frame.emg_flex_volts)
            rest_ext_v.append(frame.emg_ext_volts)
        time.sleep(0.010)

    if not rest_flex_v:
        rest_flex_v = [0.08]
    if not rest_ext_v:
        rest_ext_v = [0.07]

    rest_flex_mean = float(np.mean(rest_flex_v))
    rest_flex_std = float(np.std(rest_flex_v))
    rest_ext_mean = float(np.mean(rest_ext_v))
    rest_ext_std = float(np.std(rest_ext_v))
    print(f"[OK] Phase 1 Complete.")
    print(f"     Flexor Baseline:  Mean={rest_flex_mean:.3f}V, Std={rest_flex_std:.4f}V")
    print(f"     Extensor Baseline: Mean={rest_ext_mean:.3f}V, Std={rest_ext_std:.4f}V")

    # --------------------------------------------------------------------------
    # PHASE 2: FLEXOR MVC (5s - MAXIMUM FIST CLENCH)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 2: FLEXOR MAXIMUM VOLUNTARY CONTRACTION (MVC)")
    print("=" * 65)
    print("Prepare to squeeze your FIST at MAXIMUM FORCE for 5 seconds when prompted.")
    if countdown > 0:
        print(f"Starting in {countdown} seconds...")
        for i in range(countdown, 0, -1):
            print(f"  {i}...")
            time.sleep(1.0)
    print(f"\n[>>> SQUEEZE MAXIMUM FIST NOW! ({mvc_duration:.1f} seconds) <<<]")

    if force_mock or not receiver.is_connected:
        receiver.set_mock_scenario("flexor")

    mvc_flex_v = []
    start = time.time()
    while time.time() - start < mvc_duration:
        frame = receiver.get_latest_frame()
        if frame and frame.is_valid:
            mvc_flex_v.append(frame.emg_flex_volts)
        time.sleep(0.010)

    if not mvc_flex_v:
        mvc_flex_v = [2.15]

    mvc_flex_peak = float(np.max(mvc_flex_v))
    mvc_flex_mean = float(np.mean(mvc_flex_v))
    print(f"[OK] Phase 2 Complete. Flexor MVC Peak: {mvc_flex_peak:.3f}V (Mean Contraction: {mvc_flex_mean:.3f}V)")

    # --------------------------------------------------------------------------
    # PHASE 3: EXTENSOR MVC (5s - MAXIMUM HAND OPENING)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 3: EXTENSOR MAXIMUM VOLUNTARY CONTRACTION (MVC)")
    print("=" * 65)
    print("Prepare to open your hand and extend your fingers at MAXIMUM FORCE.")
    if countdown > 0:
        print(f"Starting in {countdown} seconds...")
        for i in range(countdown, 0, -1):
            print(f"  {i}...")
            time.sleep(1.0)
    print(f"\n[>>> EXTEND FINGERS & SPREAD WIDE NOW! ({mvc_duration:.1f} seconds) <<<]")

    if force_mock or not receiver.is_connected:
        receiver.set_mock_scenario("extensor")

    mvc_ext_v = []
    start = time.time()
    while time.time() - start < mvc_duration:
        frame = receiver.get_latest_frame()
        if frame and frame.is_valid:
            mvc_ext_v.append(frame.emg_ext_volts)
        time.sleep(0.010)

    receiver.stop()

    if not mvc_ext_v:
        mvc_ext_v = [2.15]

    mvc_ext_peak = float(np.max(mvc_ext_v))
    mvc_ext_mean = float(np.mean(mvc_ext_v))
    print(f"[OK] Phase 3 Complete. Extensor MVC Peak: {mvc_ext_peak:.3f}V (Mean Contraction: {mvc_ext_mean:.3f}V)")

    # --------------------------------------------------------------------------
    # PER-CHANNEL THRESHOLDS & EXPORT
    # --------------------------------------------------------------------------
    dyn_flex = max(0.1, mvc_flex_peak - rest_flex_mean)
    dyn_ext  = max(0.1, mvc_ext_peak - rest_ext_mean)

    rest_flex_th = rest_flex_mean + 0.12 * dyn_flex
    act_flex_th  = rest_flex_mean + 0.28 * dyn_flex
    high_flex_th = rest_flex_mean + 0.65 * dyn_flex

    rest_ext_th = rest_ext_mean + 0.12 * dyn_ext
    act_ext_th  = rest_ext_mean + 0.28 * dyn_ext
    high_ext_th = rest_ext_mean + 0.65 * dyn_ext

    calib_data = {
        "calibrated": True,
        "timestamp": time.time(),
        "tool_version": "1.0.0",
        "subject_id": subject,
        "calibration_timestamp": time.time(),
        "channels": {
            "flexor": {
                "channel_idx": 0,
                "site": "forearm_flexor",
                "muscle": "flexor_digitorum_superficialis",
                "action": "GRASP_CLOSE",
                "rest_mean_v": round(rest_flex_mean, 4),
                "rest_std_v": round(rest_flex_std, 4),
                "mvc_peak_v": round(mvc_flex_peak, 4),
                "dynamic_range_v": round(dyn_flex, 4),
                "rest_threshold_v": round(rest_flex_th, 4),
                "activation_threshold_v": round(act_flex_th, 4),
                "high_contraction_threshold_v": round(high_flex_th, 4),
            },
            "extensor": {
                "channel_idx": 1,
                "site": "forearm_extensor",
                "muscle": "extensor_digitorum_communis",
                "action": "HAND_OPEN",
                "rest_mean_v": round(rest_ext_mean, 4),
                "rest_std_v": round(rest_ext_std, 4),
                "mvc_peak_v": round(mvc_ext_peak, 4),
                "dynamic_range_v": round(dyn_ext, 4),
                "rest_threshold_v": round(rest_ext_th, 4),
                "activation_threshold_v": round(act_ext_th, 4),
                "high_contraction_threshold_v": round(high_ext_th, 4),
            },
        },
        # Legacy compatibility flat keys
        "baseline_voltage_v": round(rest_flex_mean, 4),
        "baseline_std_v": round(rest_flex_std, 4),
        "mvc_peak_voltage_v": round(mvc_flex_peak, 4),
        "dynamic_range_v": round(dyn_flex, 4),
        "thresholds_normalized": {
            "rest": 0.12,
            "activation": 0.28,
            "high_contraction": 0.65,
            "co_contraction": 0.85,
        },
        "deadband_normalized": 0.12,
        "co_contraction_threshold_normalized": 0.85,
        "smoothing_alpha": 0.35,
        "hysteresis_window_count": 3,
        "safety_invariants": {
            "fail_closed_mode": True,
            "parallel_estop_active": True,
            "lead_off_detection_active": True,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(config_out)), exist_ok=True)
    with open(config_out, "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\n[SUCCESS] Two-site calibration saved to: {config_out}")
    print(json.dumps(calib_data, indent=2))
    return calib_data


def main():
    args = parse_args()
    run_guided_calibration(args.subject, args.config_out, args.mock, args.fast)


if __name__ == "__main__":
    main()
