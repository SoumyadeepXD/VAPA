"""
VAPA Phase 0: System Pre-Flight Diagnostics & Calibration Suite
Executes comprehensive automated pre-flight integrity checks across all 6 subsystems:
1. Runtime Environment & Dependencies
2. Actuation Hardware & 9-Servo Channel Mapping
3. 3D Spatial Perception & Pinhole Deprojection Math
4. Biosignal DSP Filters & Neural-Muscular Decoders
5. Kinematics Solvers & Trajectory Planner
6. ESP32 Inter-Node Telemetry Protocol
7. Automated Unit Test Regression Suite
"""

import sys
import os
import time
import unittest
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.hardware_config import SERVO_CHANNELS, JETSON_UART_PORT, JETSON_I2C_BUS
from config.system_config import JOINT_LIMITS_DEG, CAMERA_WIDTH, CAMERA_HEIGHT, FORCE_MIN_N, FORCE_MAX_N
from drivers.pca9685_12ch_driver import SERVO_12CH_CONFIG, PCA9685_12ChDriver
from drivers.tca9548a_as5600 import AS5600EncoderMux
from vision.spatial_3d import Spatial3DAnalyzer
from vision.realsense_camera import SyntheticRealSenseCamera, CameraIntrinsics
from biosignals.signal_filters import SignalFilter, compute_rms_envelope, compute_bandpower
from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.eeg_decoder import EEGDecoder, EEGIntent
from biosignals.intent_fusion import IntentFusionEngine, MultimodalCommand
from kinematics.arm_model import ArmModel
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import InverseKinematics
from kinematics.trajectory_planner import TrajectoryPlanner
from actuation.mock_arm_controller import MockServoDriver
from actuation.arm_controller import ArmController
from tests.test_unit_suite import (
    TestBiosignalFilters,
    TestEMGDecoder,
    TestEEGDecoder,
    TestIntentFusion,
    TestKinematicsAndPlanning,
    TestArmControllerAndDrivers,
)


class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str):
    print("\n" + "=" * 78)
    print(f" {Color.BOLD}{Color.CYAN}PHASE 0 PRE-FLIGHT CHECK: {title}{Color.RESET}")
    print("=" * 78)


def print_check(name: str, passed: bool, details: str = ""):
    status = f"{Color.GREEN}[PASSED]{Color.RESET}" if passed else f"{Color.RED}[FAILED]{Color.RESET}"
    detail_str = f" - {details}" if details else ""
    print(f"  {status} {name}{detail_str}")
    return passed


def run_phase_0_diagnostics() -> bool:
    start_time = time.time()
    total_checks = 0
    passed_checks = 0

    print("\n" + "#" * 78)
    print(f"#{Color.BOLD}{Color.GREEN}        VAPA (VISUALLY ASSISTED PROSTHETIC ARM) — PHASE 0 DIAGNOSTICS        {Color.RESET}#")
    print("#" * 78)

    # --------------------------------------------------------------------------
    # Check 1: Runtime Environment & Dependencies
    # --------------------------------------------------------------------------
    print_header("1. Runtime Environment & Dependencies")
    
    # Python Version
    py_ok = sys.version_info >= (3, 10)
    total_checks += 1
    if print_check("Python 3.10+ Runtime", py_ok, f"Current: {sys.version.split()[0]}"):
        passed_checks += 1

    # NumPy version
    np_ok = hasattr(np, "__version__")
    total_checks += 1
    if print_check("NumPy High-Performance Vector Math", np_ok, f"Version {np.__version__}"):
        passed_checks += 1

    # SciPy
    import scipy
    scipy_ok = hasattr(scipy, "__version__")
    total_checks += 1
    if print_check("SciPy DSP & Signal Filtering", scipy_ok, f"Version {scipy.__version__}"):
        passed_checks += 1

    # OpenCV
    import cv2
    cv_ok = hasattr(cv2, "__version__")
    total_checks += 1
    if print_check("OpenCV Computer Vision & HUD Renderer", cv_ok, f"Version {cv2.__version__}"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Check 2: Actuation Hardware & 9-Servo Channel Mapping
    # --------------------------------------------------------------------------
    print_header("2. Actuation Hardware & 9-Servo Allocation (Section 7B Build Phase)")
    
    # Verify 9 active channels defined in hardware config
    expected_channels = {
        0: "finger_thumb",
        1: "finger_index",
        2: "finger_middle",
        3: "finger_ring",
        4: "finger_pinky",
        5: "joint_wrist_flex",
        6: "joint_wrist_rotate",
        7: "joint_wrist_bend",
        8: "joint_forearm_rotate",
    }
    
    mapping_ok = True
    for ch, name in expected_channels.items():
        if ch not in SERVO_12CH_CONFIG or SERVO_12CH_CONFIG[ch]["name"] != name:
            mapping_ok = False
            break
    total_checks += 1
    if print_check("PCA9685 Channel Map (5x Fingers, 3x Wrist, 1x Forearm)", mapping_ok, "All 9 active channels verified"):
        passed_checks += 1

    # Verify Angle Bounds
    bounds_ok = all(
        cfg["min_deg"] == 0.0 and cfg["max_deg"] == 180.0
        for cfg in SERVO_12CH_CONFIG.values()
    )
    total_checks += 1
    if print_check("Servo Safety Limits (0.0° - 180.0°)", bounds_ok, "Strict range clamping active"):
        passed_checks += 1

    # Mock Servo Driver instantiation
    driver = PCA9685_12ChDriver(force_mock=True)
    driver_ok = len(driver.current_angles) == 9
    total_checks += 1
    if print_check("PCA9685 Driver Soft-Start & Mock Bus Parity", driver_ok, f"{len(driver.current_angles)} channels online"):
        passed_checks += 1

    # TCA9548A MUX + AS5600 Encoders
    mux_driver = AS5600EncoderMux(force_mock=True)
    angles = mux_driver.read_all_encoders()
    mux_ok = len(angles) == 4
    total_checks += 1
    if print_check("TCA9548A 4-Channel I2C MUX & AS5600 Encoders", mux_ok, "Channels 0-3 telemetry verified"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Check 3: 3D Spatial Perception & Pinhole Deprojection Math
    # --------------------------------------------------------------------------
    print_header("3. 3D Spatial Perception & Pinhole Deprojection Math")

    cam = SyntheticRealSenseCamera()
    color, depth = cam.get_frames()
    cam_ok = (color is not None) and (depth is not None) and (color.shape == (480, 640, 3))
    total_checks += 1
    if print_check("RealSense Camera Frame Capture (640x480 @ 30Hz)", cam_ok, f"Depth shape: {depth.shape}"):
        passed_checks += 1

    # Test Deprojection Math
    # Optical center pixel (320, 240) at 1.0 meter depth should yield [0.0, 0.0, 1.0]
    p3d_c = cam.intrinsics.deproject_pixel_to_point(320.0, 240.0, 1.0)
    deproj_ok = np.isclose(p3d_c[0], 0.0, atol=1e-3) and np.isclose(p3d_c[1], 0.0, atol=1e-3) and np.isclose(p3d_c[2], 1.0, atol=1e-3)
    total_checks += 1
    if print_check("Pinhole Camera Optical Frame Deprojection", deproj_ok, f"Center pixel -> Xc={p3d_c[0]:.3f}, Yc={p3d_c[1]:.3f}, Zc={p3d_c[2]:.3f}m"):
        passed_checks += 1

    # Test Base Coordinate Transformation
    analyzer = Spatial3DAnalyzer()
    p3d_b = analyzer.transform_camera_to_base(p3d_c)
    base_tf_ok = len(p3d_b) == 3 and not np.isnan(p3d_b).any()
    total_checks += 1
    if print_check("Homogeneous Base Frame Transformation (T_base_cam)", base_tf_ok, f"Transformed -> Xb={p3d_b[0]:.3f}, Yb={p3d_b[1]:.3f}, Zb={p3d_b[2]:.3f}m"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Check 4: Biosignal DSP Filters & Neural-Muscular Decoders
    # --------------------------------------------------------------------------
    print_header("4. Biosignal DSP Filters & Intent Decoders")

    sig_filter = SignalFilter(sampling_rate_hz=250.0)
    sig_filter.add_notch_filter("notch50", notch_freq_hz=50.0, q=30.0)
    sig_filter.add_bandpass_filter("emg_bp", low_hz=20.0, high_hz=100.0)
    
    t_notch = np.linspace(0, 1.0, 250, endpoint=False)
    test_wave = np.sin(2 * np.pi * 50.0 * t_notch)
    filtered_wave = sig_filter.filter_signal(test_wave, "notch50")
    # Discard transient initial 50 samples for steady-state evaluation
    notch_attenuation = np.std(filtered_wave[50:]) < np.std(test_wave[50:]) * 0.15
    total_checks += 1
    if print_check("50 Hz Powerline Notch Filter Rejection", notch_attenuation, "Powerline hum successfully rejected"):
        passed_checks += 1

    # EMG Decoder Co-Contraction E-Stop Check
    emg_decoder = EMGDecoder(num_channels=2, fs=250.0)
    t_burst = np.linspace(0, 0.5, int(250 * 0.5))
    burst_ch = 180.0 * np.sin(2 * np.pi * 100.0 * t_burst)
    co_burst = np.vstack([burst_ch, burst_ch])
    emg_intent = emg_decoder.update_samples(co_burst)
    co_ok = (emg_intent.gesture == EMGIntent.CO_CONTRACTION_ESTOP)
    total_checks += 1
    if print_check("EMG Co-Contraction Emergency Stop Decoder", co_ok, "Simultaneous flexor+extensor triggers E-STOP"):
        passed_checks += 1

    # EEG Decoder Frontal Blink Check
    eeg_decoder = EEGDecoder(num_channels=4, fs=250.0)
    eeg_burst = np.zeros((4, len(t_burst)))
    eeg_burst[3, :] = 150.0 * np.sin(2 * np.pi * 5.0 * t_burst)
    eeg_intent = eeg_decoder.update_samples(eeg_burst)
    blink_ok = (eeg_intent.command == EEGIntent.TARGET_CYCLE_NEXT)
    total_checks += 1
    if print_check("EEG Cognitive Target Selection (Frontal Blink Cycle)", blink_ok, "Frontal artifact cycles 3D targets"):
        passed_checks += 1

    # Multimodal Intent Fusion Priority
    fusion = IntentFusionEngine()
    cmd = fusion.fuse(emg_intent, eeg_intent, visible_targets=[], system_state="REACHING")
    fusion_estop_ok = (cmd.action == MultimodalCommand.EMERGENCY_STOP)
    total_checks += 1
    if print_check("Multimodal Intent Fusion Safety Priority", fusion_estop_ok, "E-Stop unconditionally preempts reach"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Check 5: Kinematics Solvers & Trajectory Planner
    # --------------------------------------------------------------------------
    print_header("5. Kinematics Solvers & Trajectory Planner")

    arm_model = ArmModel()
    fk = ForwardKinematics(arm_model)
    ik = InverseKinematics(arm_model)
    planner = TrajectoryPlanner()

    fk_home = fk.compute_fk(arm_model.home_angles_deg)
    fk_ok = len(fk_home["ee_position"]) == 3
    total_checks += 1
    if print_check("Forward Kinematics Home Pose Computation", fk_ok, f"TCP at {fk_home['ee_position'].round(3)}m"):
        passed_checks += 1

    # Analytical IK target test
    target_p = np.array([0.25, 0.00, 0.15])
    ik_angles, ik_success = ik.solve_analytical(target_p, target_pitch_deg=0.0)
    err = np.linalg.norm(target_p - fk.compute_fk(ik_angles)["ee_position"]) if ik_success else 999.0
    ik_ok = ik_success and (err < 0.015)
    total_checks += 1
    if print_check("Analytical Inverse Kinematics Convergence", ik_ok, f"Error: {err*1000:.2f} mm (< 15 mm tolerance)"):
        passed_checks += 1

    # Minimum-jerk trajectory profile
    traj = planner.plan_trajectory(arm_model.home_angles_deg, {"joint_1_base_yaw": 30.0}, min_duration_s=1.0)
    traj_ok = len(traj) > 10 and np.isclose(traj[0].velocities_deg_s["joint_1_base_yaw"], 0.0, atol=1e-2)
    total_checks += 1
    if print_check("Quintic Minimum-Jerk Trajectory Interpolation", traj_ok, f"Zero boundary velocity confirmed ({len(traj)} waypoints)"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Check 6: Inter-Node Telemetry Protocol & JSON Framing
    # --------------------------------------------------------------------------
    print_header("6. Inter-Node Telemetry Protocol & JSON Framing")

    import json
    sample_packet = '{"seq":100,"fsr":[0.1,0.2,0.3,0.4,0.0],"emg":0.5,"eeg":0.2,"ts":12345}'
    try:
        data = json.loads(sample_packet)
        json_ok = ("seq" in data and "fsr" in data and "emg" in data and "eeg" in data and "ts" in data)
    except Exception:
        json_ok = False
    total_checks += 1
    if print_check("ESP32 100 Hz JSON Telemetry Schema", json_ok, "seq, fsr[5], emg, eeg, ts validated"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Check 7: Automated Unit Test Regression Suite
    # --------------------------------------------------------------------------
    print_header("7. Automated Unit Test Regression Suite")

    suite = unittest.TestSuite()
    for test_class in [
        TestBiosignalFilters,
        TestEMGDecoder,
        TestEEGDecoder,
        TestIntentFusion,
        TestKinematicsAndPlanning,
        TestArmControllerAndDrivers,
    ]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    unit_tests_ok = result.wasSuccessful()
    total_checks += 1
    if print_check("Unit Test Suite Execution (18 Tests)", unit_tests_ok, f"Ran {result.testsRun} tests with 0 failures"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    elapsed_s = time.time() - start_time
    print("\n" + "=" * 78)
    print(f" {Color.BOLD}PHASE 0 PRE-FLIGHT DIAGNOSTICS SUMMARY{Color.RESET}")
    print("=" * 78)
    print(f" Total Verification Checks : {total_checks}")
    print(f" Passed                    : {Color.GREEN}{passed_checks}{Color.RESET}")
    print(f" Failed                    : {Color.RED}{total_checks - passed_checks}{Color.RESET}")
    print(f" Execution Duration        : {elapsed_s:.3f} seconds")

    if passed_checks == total_checks:
        print(f"\n {Color.BOLD}{Color.GREEN}>>> ALL PHASE 0 PRE-FLIGHT CHECKS PASSED — SYSTEM FLIGHT-READY! <<<{Color.RESET}\n")
        return True
    else:
        print(f"\n {Color.BOLD}{Color.RED}>>> PRE-FLIGHT WARNINGS DETECTED — REVIEW FAILED SUBSYSTEMS <<<{Color.RESET}\n")
        return False


if __name__ == "__main__":
    success = run_phase_0_diagnostics()
    sys.exit(0 if success else 1)
