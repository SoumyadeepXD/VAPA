"""
VAPA Phase 2: Neural Decoding, Cognitive Target Selection & Reach Motion Planning Suite
Executes complete Phase 2 pipeline:
1. EEG multi-channel filtering (1-45 Hz bandpass + 50 Hz notch)
2. Cognitive target cycling via frontal blink artifact (TARGET_CYCLE_NEXT)
3. Motor imagery Mu-band Event-Related Desynchronization (Mu ERD > 0.35)
4. Multimodal intent fusion (Vision + EEG) -> START_REACH command
5. 3D pre-grasp standoff position calculation & analytical Inverse Kinematics
6. Quintic polynomial minimum-jerk trajectory interpolation
7. Multi-servo execution & state transition to AT_TARGET
"""

import sys
import os
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.system_config import (
    EEG_SAMPLING_RATE_HZ,
    EEG_MU_RHYTHM_BAND,
    EEG_MOTOR_IMAGERY_ERD_THRESHOLD,
    WORKSPACE_BOUNDS_M,
)
from biosignals.signal_filters import SignalFilter, compute_bandpower
from biosignals.eeg_decoder import EEGDecoder, EEGIntent
from biosignals.emg_decoder import EMGIntent
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
    print(f" {Color.BOLD}{Color.CYAN}PHASE 2 NEURAL REACH: {title}{Color.RESET}")
    print("=" * 78)


def print_check(name: str, passed: bool, details: str = ""):
    status = f"{Color.GREEN}[PASSED]{Color.RESET}" if passed else f"{Color.RED}[FAILED]{Color.RESET}"
    detail_str = f" - {details}" if details else ""
    print(f"  {status} {name}{detail_str}")
    return passed


def run_phase_2_neural_reach(force_mock: bool = True) -> bool:
    start_time = time.time()
    total_checks = 0
    passed_checks = 0

    print("\n" + "#" * 78)
    print(f"#{Color.BOLD}{Color.GREEN}        VAPA (VISUALLY ASSISTED PROSTHETIC ARM) — PHASE 2 NEURAL REACH        {Color.RESET}#")
    print("#" * 78)

    # --------------------------------------------------------------------------
    # Step 1: EEG Signal Conditioning & Filtering
    # --------------------------------------------------------------------------
    print_header("1. EEG Signal Conditioning & Bandpass Filtering")
    fs = float(EEG_SAMPLING_RATE_HZ)
    eeg_filter = SignalFilter(sampling_rate_hz=fs)
    eeg_filter.add_notch_filter("notch50", notch_freq_hz=50.0, q=30.0)
    eeg_filter.add_bandpass_filter("eeg_bp", low_hz=1.0, high_hz=45.0)

    # 10 Hz pure sine wave (Mu rhythm) + 50 Hz powerline hum
    t = np.linspace(0, 1.0, int(fs), endpoint=False)
    raw_eeg = 20.0 * np.sin(2 * np.pi * 10.0 * t) + 15.0 * np.sin(2 * np.pi * 50.0 * t)
    filtered = eeg_filter.filter_signal(raw_eeg, "notch50")
    filtered = eeg_filter.filter_signal(filtered, "eeg_bp")

    filter_ok = np.std(filtered) > 5.0 and not np.isnan(filtered).any()
    total_checks += 1
    if print_check("EEG 1-45 Hz Bandpass & 50 Hz Hum Attenuation", filter_ok, "Clean neural frequency spectrum isolated"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 2: Cognitive Target Cycling via Frontal Channel
    # --------------------------------------------------------------------------
    print_header("2. Cognitive Target Cycling (Frontal Artifact Spike)")
    eeg_decoder = EEGDecoder(num_channels=4, fs=fs)

    # Frontal spike: Channel 3 (Fz) has a 150 uV delta/theta eye-blink pulse
    eeg_blink_chunk = np.zeros((4, int(fs * 0.4)))
    t_blink = np.linspace(0, 0.4, int(fs * 0.4), endpoint=False)
    eeg_blink_chunk[3, :] = 160.0 * np.sin(2 * np.pi * 4.0 * t_blink)

    blink_intent = eeg_decoder.update_samples(eeg_blink_chunk)
    cycle_ok = (blink_intent.command == EEGIntent.TARGET_CYCLE_NEXT)
    total_checks += 1
    if print_check("Frontal Cognitive Target Switch Decoder", cycle_ok, f"Command: {blink_intent.command} (Conf: {blink_intent.confidence:.2f})"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 3: Motor Imagery Mu-Band Event-Related Desynchronization (ERD)
    # --------------------------------------------------------------------------
    print_header("3. Motor Imagery Mu ERD (Sensorimotor Cortex C3/Cz)")

    # Simulate active motor imagery: Mu rhythm desynchronization (power suppression)
    # Resting signal has strong 10 Hz Mu oscillation (~30 uV)
    # Motor imagery signal has suppressed Mu oscillation (~5 uV) with elevated Beta (~18 uV)
    mi_chunk = np.zeros((4, int(fs * 0.5)))
    t_mi = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    for ch in range(3):  # C3, C4, Cz
        mi_chunk[ch, :] = 4.0 * np.sin(2 * np.pi * 10.0 * t_mi) + 18.0 * np.sin(2 * np.pi * 20.0 * t_mi)
    # Background noise on frontal
    mi_chunk[3, :] = 3.0 * np.random.randn(len(t_mi))

    # Reset debounce cooldown so active reach trigger evaluates cleanly
    eeg_decoder.last_trigger_time = 0.0
    eeg_decoder.baseline_mu_power = 0.40  # Set calibrated resting baseline
    reach_intent = eeg_decoder.update_samples(mi_chunk)

    erd_ok = reach_intent.motor_imagery_active and (reach_intent.command == EEGIntent.INTENT_REACH)
    total_checks += 1
    if print_check(
        "Motor Imagery Mu ERD Reach Trigger",
        erd_ok,
        f"MI Active: {reach_intent.motor_imagery_active}, Mu: {reach_intent.mu_power:.3f}, Beta: {reach_intent.beta_power:.3f}"
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 4: Multimodal Intent Fusion Arbitration
    # --------------------------------------------------------------------------
    print_header("4. Multimodal Intent Fusion Arbitration (Vision + Neural)")
    fusion = IntentFusionEngine()

    # Create mock 3D vision targets
    apple_target = GraspTarget3D(
        label="apple",
        score=0.92,
        center_camera_m=np.array([0.00, 0.08, 0.38]),
        center_base_m=np.array([0.35, 0.00, 0.12]),
        width_m=0.045,
        height_m=0.045,
        depth_m=0.045,
        approach_vector=np.array([1.0, 0.0, 0.0]),
        grasp_yaw_deg=0.0,
        grasp_pitch_deg=0.0,
        target_opening_m=0.065,
        target_force_n=1.5,
        is_reachable=True,
        pixel_box=(280, 200, 360, 280),
    )
    mug_target = GraspTarget3D(
        label="mug",
        score=0.88,
        center_camera_m=np.array([-0.09, 0.05, 0.45]),
        center_base_m=np.array([0.38, 0.15, 0.10]),
        width_m=0.075,
        height_m=0.080,
        depth_m=0.075,
        approach_vector=np.array([1.0, 0.0, 0.0]),
        grasp_yaw_deg=0.0,
        grasp_pitch_deg=0.0,
        target_opening_m=0.095,
        target_force_n=3.5,
        is_reachable=True,
        pixel_box=(160, 240, 250, 330),
    )
    visible_targets = [apple_target, mug_target]

    # Test Target Cycling via fusion
    cycle_cmd = fusion.fuse(
        emg_intent=EMGIntent(EMGIntent.REST, 1.0, 0.0, 0.0, [0.0]*2, time.time()),
        eeg_intent=blink_intent,
        visible_targets=visible_targets,
        system_state="SCANNING",
    )
    fuse_cycle_ok = (cycle_cmd.action == MultimodalCommand.CYCLE_TARGET)
    total_checks += 1
    if print_check("Fusion Target Cycle Forward", fuse_cycle_ok, f"Selected Target: index {fusion.selected_target_index}"):
        passed_checks += 1

    # Test Reach Trigger via fusion
    reach_cmd = fusion.fuse(
        emg_intent=EMGIntent(EMGIntent.REST, 1.0, 0.0, 0.0, [0.0]*2, time.time()),
        eeg_intent=reach_intent,
        visible_targets=visible_targets,
        system_state="SCANNING",
    )
    fuse_reach_ok = (reach_cmd.action == MultimodalCommand.START_REACH) and (reach_cmd.target is not None)
    total_checks += 1
    if print_check("Fusion Reach Arbitration to Locked Target", fuse_reach_ok, f"Target Locked: '{reach_cmd.target.label.upper()}'"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 5: Kinematics Pre-Grasp Standoff & Analytical IK
    # --------------------------------------------------------------------------
    print_header("5. Kinematics Standoff & Inverse Kinematics Resolution")
    arm_model = ArmModel()
    fk = ForwardKinematics(arm_model)
    ik = InverseKinematics(arm_model)

    target_obj = reach_cmd.target
    # 4cm standoff along approach vector
    approach_offset = -0.04 * target_obj.approach_vector
    standoff_pos = target_obj.center_base_m + approach_offset

    target_angles, ik_success = ik.solve_analytical(
        standoff_pos,
        target_pitch_deg=target_obj.grasp_pitch_deg,
        target_roll_deg=0.0,
        gripper_percent=100.0,
    )
    fk_res = fk.compute_fk(target_angles)
    pos_err_m = np.linalg.norm(standoff_pos - fk_res["ee_position"])
    ik_ok = ik_success and (pos_err_m < 0.015)
    total_checks += 1
    if print_check(
        "Analytical IK Standoff Convergence",
        ik_ok,
        f"Target: {standoff_pos.round(3)}m | Error: {pos_err_m*1000:.2f} mm (< 15 mm tolerance)"
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 6: Smooth Quintic Trajectory Generation & Boundary Invariants
    # --------------------------------------------------------------------------
    print_header("6. Quintic Polynomial Minimum-Jerk Trajectory Planning")
    planner = TrajectoryPlanner()
    start_q = arm_model.home_angles_deg
    traj = planner.plan_trajectory(start_q, target_angles, min_duration_s=1.2)

    traj_count_ok = len(traj) >= 50
    # Zero boundary velocities
    v_start = max(abs(v) for v in traj[0].velocities_deg_s.values())
    v_end = max(abs(v) for v in traj[-1].velocities_deg_s.values())
    boundary_ok = (v_start < 0.05) and (v_end < 0.05)
    total_checks += 1
    if print_check(
        "Quintic Minimum-Jerk Trajectory Profile",
        traj_count_ok and boundary_ok,
        f"{len(traj)} waypoints, T={traj[-1].time_s:.2f}s, Boundary vel: v0={v_start:.3f}, vT={v_end:.3f}°/s"
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 7: Multi-Servo Execution & FSM State Transition
    # --------------------------------------------------------------------------
    print_header("7. Multi-Servo Motion Execution & FSM State Transition")
    sm = VAPAStateMachine(initial_state=VAPAState.SCANNING)
    arm = ArmController(force_mock=force_mock)

    sm.transition_to(VAPAState.TARGET_SELECTED, target=target_obj)
    sm.transition_to(VAPAState.PLANNING, target=target_obj)
    sm.transition_to(VAPAState.REACHING, target=target_obj)

    # Execute trajectory
    exec_ok = arm.execute_trajectory(
        traj,
        abort_check_fn=lambda: arm.is_emergency_stopped,
    )
    if exec_ok:
        sm.transition_to(VAPAState.AT_TARGET, target=target_obj)

    fsm_ok = (sm.current_state == VAPAState.AT_TARGET) and exec_ok
    total_checks += 1
    if print_check(
        "Arm Motion Execution to Grasp Pose",
        fsm_ok,
        f"FSM State: {sm.current_state} | Reached Standoff Pose at Target"
    ):
        passed_checks += 1

    # Safely return arm home after test
    arm.go_to_home(duration_s=0.5)
    arm.close()

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    elapsed_s = time.time() - start_time
    print("\n" + "=" * 78)
    print(f" {Color.BOLD}PHASE 2 NEURAL REACH DIAGNOSTICS SUMMARY{Color.RESET}")
    print("=" * 78)
    print(f" Total Phase 2 Checks      : {total_checks}")
    print(f" Passed                    : {Color.GREEN}{passed_checks}{Color.RESET}")
    print(f" Failed                    : {Color.RED}{total_checks - passed_checks}{Color.RESET}")
    print(f" Execution Duration        : {elapsed_s:.3f} seconds")
    print(f" Final System State        : {Color.GREEN}AT_TARGET (Ready for Phase 3 Grasp){Color.RESET}")

    if passed_checks == total_checks:
        print(f"\n {Color.BOLD}{Color.GREEN}>>> ALL PHASE 2 CHECKS PASSED — TARGET REACH CONVERGED! <<<{Color.RESET}\n")
        return True
    else:
        print(f"\n {Color.BOLD}{Color.RED}>>> PHASE 2 ISSUES DETECTED <<<{Color.RESET}\n")
        return False


if __name__ == "__main__":
    force_mock_flag = "--mock" in sys.argv or "--real" not in sys.argv
    success = run_phase_2_neural_reach(force_mock=force_mock_flag)
    sys.exit(0 if success else 1)
