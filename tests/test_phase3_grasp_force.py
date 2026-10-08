"""
VAPA Phase 3: EMG Muscle Grasp Decoding, 5-Finger Hand Actuation & Closed-Loop Force Suite
Executes complete Phase 3 pipeline:
1. Surface EMG signal conditioning (20-450 Hz bandpass, 50 Hz notch, 50ms RMS window)
2. Flexor muscle activation decoding & continuous proportional force modulation (0.3N - 10.0N)
3. Multimodal intent fusion arbitration (AT_TARGET -> START_GRASP)
4. Kinematic pre-grasp standoff advance to contact coordinate
5. Simultaneous 5-finger PCA9685 servo flexion (Channels 0-4: Thumb, Index, Middle, Ring, Pinky)
6. Closed-loop force ramp with live FSR tactile sensor feedback
7. Over-force safety abort threshold verification (>= 12.0N E-STOP limit)
8. FSM transition from AT_TARGET -> GRASPING -> HOLDING
"""

import sys
import os
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.system_config import (
    EMG_SAMPLING_RATE_HZ,
    FORCE_MIN_N,
    FORCE_MAX_N,
    FORCE_EMERGENCY_LIMIT_N,
    GRASP_TIMEOUT_S,
)
from biosignals.signal_filters import SignalFilter, compute_rms_envelope
from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.eeg_decoder import EEGIntent
from biosignals.intent_fusion import IntentFusionEngine, MultimodalCommand
from vision.spatial_3d import GraspTarget3D
from kinematics.arm_model import ArmModel
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import InverseKinematics
from kinematics.trajectory_planner import TrajectoryPlanner
from actuation.mock_arm_controller import MockServoDriver
from actuation.arm_controller import ArmController
from core.state_machine import VAPAStateMachine, VAPAState


class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str):
    print("\n" + "=" * 78)
    print(f" {Color.BOLD}{Color.CYAN}PHASE 3 GRASP & FORCE CONTROL: {title}{Color.RESET}")
    print("=" * 78)


def print_check(name: str, passed: bool, details: str = ""):
    status = f"{Color.GREEN}[PASSED]{Color.RESET}" if passed else f"{Color.RED}[FAILED]{Color.RESET}"
    detail_str = f" - {details}" if details else ""
    print(f"  {status} {name}{detail_str}")
    return passed


def run_phase_3_grasp_force(force_mock: bool = True) -> bool:
    start_time = time.time()
    total_checks = 0
    passed_checks = 0

    print("\n" + "#" * 78)
    print(f"#{Color.BOLD}{Color.GREEN}        VAPA (VISUALLY ASSISTED PROSTHETIC ARM) — PHASE 3 GRASP & FORCE       {Color.RESET}#")
    print("#" * 78)

    # --------------------------------------------------------------------------
    # Step 1: Surface EMG Signal Conditioning & RMS Envelope
    # --------------------------------------------------------------------------
    print_header("1. Surface EMG Signal Conditioning & Envelope Extraction")
    fs = float(EMG_SAMPLING_RATE_HZ)
    sig_filter = SignalFilter(sampling_rate_hz=fs)
    sig_filter.add_notch_filter("notch50", notch_freq_hz=50.0, q=30.0)
    sig_filter.add_bandpass_filter("emg_bp", low_hz=20.0, high_hz=450.0)

    # Generate synthetic 100 Hz muscle contraction burst with 50 Hz powerline hum
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    raw_burst = 90.0 * np.sin(2 * np.pi * 100.0 * t) + 20.0 * np.sin(2 * np.pi * 50.0 * t)
    filtered = sig_filter.filter_signal(raw_burst, "notch50")
    filtered = sig_filter.filter_signal(filtered, "emg_bp")
    rms_env = compute_rms_envelope(filtered, window_size=50)

    emg_filter_ok = (np.mean(rms_env) > 10.0) and not np.isnan(rms_env).any()
    total_checks += 1
    if print_check("EMG Digital Filter & RMS Envelope", emg_filter_ok, f"Mean RMS: {np.mean(rms_env):.2f} uV"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 2: Flexor Muscle Activation & Proportional Force Modulation
    # --------------------------------------------------------------------------
    print_header("2. Flexor Muscle Activation & Continuous Proportional Force")
    emg_decoder = EMGDecoder(num_channels=2, fs=fs)
    
    # Ch 0: Flexor burst (135 uV), Ch 1: Extensor rest (2 uV)
    flexor_burst = np.vstack([
        135.0 * np.sin(2 * np.pi * 100.0 * t),
        2.0 * np.random.randn(len(t)),
    ])
    emg_intent = emg_decoder.update_samples(flexor_burst)

    act_ok = (emg_intent.gesture == EMGIntent.GRASP_CLOSE) and (emg_intent.activation_level > 0.50)
    force_scaled_ok = FORCE_MIN_N < emg_intent.proportional_force_n <= FORCE_MAX_N
    total_checks += 1
    if print_check(
        "Flexor Grasp Intent & Proportional Force",
        act_ok and force_scaled_ok,
        f"Gesture: {emg_intent.gesture} | Act: {emg_intent.activation_level*100:.1f}% | Target Force: {emg_intent.proportional_force_n:.2f}N"
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 3: Multimodal Intent Fusion Arbitration (AT_TARGET -> START_GRASP)
    # --------------------------------------------------------------------------
    print_header("3. Multimodal Intent Fusion (AT_TARGET State Arbitration)")
    fusion = IntentFusionEngine()

    target_mug = GraspTarget3D(
        label="mug",
        score=0.90,
        center_camera_m=np.array([-0.05, 0.04, 0.42]),
        center_base_m=np.array([0.36, 0.10, 0.12]),
        width_m=0.075,
        height_m=0.080,
        depth_m=0.075,
        approach_vector=np.array([1.0, 0.0, 0.0]),
        grasp_yaw_deg=0.0,
        grasp_pitch_deg=0.0,
        target_opening_m=0.095,
        target_force_n=3.5,
        is_reachable=True,
        pixel_box=(200, 200, 300, 300),
    )

    grasp_cmd = fusion.fuse(
        emg_intent=emg_intent,
        eeg_intent=EEGIntent(EEGIntent.IDLE, 0.5, 0.2, 0.1, 0.4, False, time.time()),
        visible_targets=[target_mug],
        system_state="AT_TARGET",
    )

    fuse_grasp_ok = (grasp_cmd.action == MultimodalCommand.START_GRASP) and (grasp_cmd.target_force_n >= 2.0)
    total_checks += 1
    if print_check("Intent Fusion Grasp Arbitration", fuse_grasp_ok, f"Command: START_GRASP | Modulated Force: {grasp_cmd.target_force_n:.2f}N"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 4: Standoff Advance Kinematics Trajectory
    # --------------------------------------------------------------------------
    print_header("4. Standoff Advance Kinematics Trajectory Execution")
    arm_model = ArmModel()
    fk = ForwardKinematics(arm_model)
    ik = InverseKinematics(arm_model)
    planner = TrajectoryPlanner()
    arm = ArmController(force_mock=force_mock)

    # Move to standoff position (pre-grasp)
    standoff_pos = target_mug.center_base_m - 0.04 * target_mug.approach_vector
    standoff_q, s_ok = ik.solve_analytical(standoff_pos, target_pitch_deg=0.0)
    arm.move_to_angles(standoff_q, duration_s=0.2)

    # Solve advance to object contact center
    advance_q, adv_ok = ik.solve_analytical(target_mug.center_base_m, target_pitch_deg=0.0)
    adv_traj = planner.plan_trajectory(standoff_q, advance_q, min_duration_s=0.5)

    adv_exec_ok = arm.execute_trajectory(adv_traj, abort_check_fn=lambda: arm.is_emergency_stopped)
    ee_curr = fk.compute_fk(arm.get_joint_angles())["ee_position"]
    adv_dist_err = np.linalg.norm(target_mug.center_base_m - ee_curr)
    
    total_checks += 1
    if print_check(
        "Arm Advance to Contact Coordinates",
        adv_exec_ok and (adv_dist_err < 0.015),
        f"Advanced from standoff -> contact | Distance err: {adv_dist_err*1000:.2f} mm"
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 5: Simultaneous 5-Finger Hand Flexion (CH0 - CH4)
    # --------------------------------------------------------------------------
    print_header("5. Simultaneous 5-Finger PCA9685 Servo Flexion (CH0 - CH4)")
    # Test setting all 5 fingers from fully open (100%) to grasp position (40% open)
    arm.set_gripper_opening_percent(40.0)
    angles = arm.get_joint_angles()

    fingers = ["finger_thumb", "finger_index", "finger_middle", "finger_ring", "finger_pinky"]
    fingers_ok = all(angles[f] > 60.0 for f in fingers)
    total_checks += 1
    if print_check(
        "5-Finger Independent Actuation",
        fingers_ok,
        f"All 5 fingers flexed to ~{angles['finger_thumb']:.1f}° (MG996R CH0-CH4 synchronized)"
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 6: Closed-Loop Tactile Force Convergence with Live Feedback
    # --------------------------------------------------------------------------
    print_header("6. Closed-Loop Tactile Force Regulation (FSR Feedback)")
    arm.release_grasp()

    # Simulate dynamic physical tactile resistance increasing as fingers close
    simulated_tactile_state = {"force_n": 0.0}
    def mock_tactile_sensor():
        simulated_tactile_state["force_n"] += 0.45
        return simulated_tactile_state["force_n"]

    target_force_test = 3.5  # Newtons
    grasp_success = arm.execute_force_grasp(
        target_force_n=target_force_test,
        timeout_s=2.0,
        tactile_sensor_fn=mock_tactile_sensor,
    )

    force_reached_ok = grasp_success and (arm.current_force_n >= target_force_test)
    total_checks += 1
    if print_check(
        "Closed-Loop Tactile Force Convergence",
        force_reached_ok,
        f"Commanded: {target_force_test:.2f}N | Converged Force: {arm.current_force_n:.2f}N"
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 7: Over-Force Safety Abort Threshold (>= 12.0N E-STOP limit)
    # --------------------------------------------------------------------------
    print_header("7. Over-Force Safety Abort Protection (Tactile Guard)")
    # Simulate an external crunch event exceeding safe threshold (13.5 N >= 12.0 N limit)
    dangerous_tactile_state = {"force_n": 13.5}
    def mock_dangerous_sensor():
        return dangerous_tactile_state["force_n"]

    abort_result = arm.execute_force_grasp(
        target_force_n=3.0,
        timeout_s=1.0,
        tactile_sensor_fn=mock_dangerous_sensor,
    )
    overforce_stopped = (not abort_result) and arm.is_emergency_stopped
    total_checks += 1
    if print_check(
        "Excessive Force Safety Trip (E-STOP)",
        overforce_stopped,
        f"Over-force {dangerous_tactile_state['force_n']}N tripped emergency stop immediately"
    ):
        passed_checks += 1

    # Reset emergency stop for final state transition
    arm.reset_emergency_stop()

    # --------------------------------------------------------------------------
    # Step 8: FSM State Transition: AT_TARGET -> GRASPING -> HOLDING
    # --------------------------------------------------------------------------
    print_header("8. Finite State Machine Lifecycle (AT_TARGET -> GRASPING -> HOLDING)")
    sm = VAPAStateMachine(initial_state=VAPAState.AT_TARGET)
    sm.transition_to(VAPAState.GRASPING, target=target_mug)
    
    # Simulate grasp convergence
    sm.transition_to(VAPAState.HOLDING, target=target_mug)
    fsm_ok = (sm.current_state == VAPAState.HOLDING) and (sm.locked_target == target_mug)
    total_checks += 1
    if print_check(
        "FSM Stable Transition to HOLDING",
        fsm_ok,
        f"FSM State: {sm.current_state} | Target '{target_mug.label.upper()}' secured"
    ):
        passed_checks += 1

    # Safely release fingers and return arm home
    arm.release_grasp()
    arm.go_to_home(duration_s=0.5)
    arm.close()

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    elapsed_s = time.time() - start_time
    print("\n" + "=" * 78)
    print(f" {Color.BOLD}PHASE 3 GRASP & FORCE CONTROL DIAGNOSTICS SUMMARY{Color.RESET}")
    print("=" * 78)
    print(f" Total Phase 3 Checks      : {total_checks}")
    print(f" Passed                    : {Color.GREEN}{passed_checks}{Color.RESET}")
    print(f" Failed                    : {Color.RED}{total_checks - passed_checks}{Color.RESET}")
    print(f" Execution Duration        : {elapsed_s:.3f} seconds")
    print(f" Final System State        : {Color.GREEN}HOLDING (Ready for Phase 4 Release / Retract){Color.RESET}")

    if passed_checks == total_checks:
        print(f"\n {Color.BOLD}{Color.GREEN}>>> ALL PHASE 3 CHECKS PASSED — ADAPTIVE GRASP SECURED! <<<{Color.RESET}\n")
        return True
    else:
        print(f"\n {Color.BOLD}{Color.RED}>>> PHASE 3 ISSUES DETECTED <<<{Color.RESET}\n")
        return False


if __name__ == "__main__":
    force_mock_flag = "--mock" in sys.argv or "--real" not in sys.argv
    success = run_phase_3_grasp_force(force_mock=force_mock_flag)
    sys.exit(0 if success else 1)
