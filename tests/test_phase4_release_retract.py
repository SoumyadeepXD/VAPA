"""
VAPA Phase 4: Object Holding, Extensor Release Decoding & Retraction to Home Suite
Executes complete Phase 4 pipeline:
1. Holding state retention & tactile grip modulation (MODULATE_FORCE stability)
2. Surface EMG extensor signal conditioning & HAND_OPEN decoding (Ch 1 Extensor Digitorum)
3. Multimodal intent fusion release arbitration & neural disturbance rejection (HOLDING -> RELEASE_GRIP)
4. Synchronized 5-finger PCA9685 servo extension (MG996R CH0-CH4 to 0.0°, tactile force to 0.0N)
5. Clean standoff clearance disengagement kinematics (analytical IK standoff)
6. Quintic polynomial minimum-jerk retraction trajectory planning (zero boundary velocity)
7. Multi-servo retraction trajectory execution & home pose convergence (DEFAULT_BUILD_PHASE_HOME)
8. End-to-end FSM lifecycle completion (HOLDING -> RELEASING -> RETRACTING -> IDLE)
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
from actuation.arm_controller import ArmController, DEFAULT_BUILD_PHASE_HOME
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
    print(f" {Color.BOLD}{Color.CYAN}PHASE 4 RELEASE & RETRACT: {title}{Color.RESET}")
    print("=" * 78)


def print_check(name: str, passed: bool, details: str = ""):
    status = f"{Color.GREEN}[PASSED]{Color.RESET}" if passed else f"{Color.RED}[FAILED]{Color.RESET}"
    detail_str = f" - {details}" if details else ""
    print(f"  {status} {name}{detail_str}")
    return passed


def run_phase_4_release_retract(force_mock: bool = True) -> bool:
    start_time = time.time()
    total_checks = 0
    passed_checks = 0

    print("\n" + "#" * 78)
    print(f"#{Color.BOLD}{Color.GREEN}     VAPA (VISUALLY ASSISTED PROSTHETIC ARM) — PHASE 4 RELEASE & RETRACT      {Color.RESET}#")
    print("#" * 78)

    # --------------------------------------------------------------------------
    # Step 1: Holding State Retention & Proportional Grip Modulation
    # --------------------------------------------------------------------------
    print_header("1. Holding State Retention & Grip Force Modulation")
    fusion = IntentFusionEngine()

    target_mug = GraspTarget3D(
        label="mug",
        score=0.91,
        center_camera_m=np.array([-0.05, 0.04, 0.42]),
        center_base_m=np.array([0.36, 0.10, 0.12]),
        width_m=0.075,
        height_m=0.080,
        depth_m=0.075,
        approach_vector=np.array([1.0, 0.0, 0.0]),
        grasp_yaw_deg=0.0,
        grasp_pitch_deg=0.0,
        target_opening_m=0.095,
        target_force_n=4.0,
        is_reachable=True,
        pixel_box=(200, 200, 300, 300),
    )

    # In HOLDING, flexor muscle contraction modulates grip firmness without dropping
    mod_flexor_intent = EMGIntent(
        gesture=EMGIntent.GRASP_CLOSE,
        confidence=0.92,
        proportional_force_n=5.2,
        activation_level=0.55,
        channel_rms=[65.0, 5.0],
        timestamp=time.time(),
    )
    mod_cmd = fusion.fuse(
        emg_intent=mod_flexor_intent,
        eeg_intent=EEGIntent(EEGIntent.IDLE, 0.5, 0.2, 0.1, 0.4, False, time.time()),
        visible_targets=[target_mug],
        system_state="HOLDING",
    )

    holding_mod_ok = (mod_cmd.action == MultimodalCommand.MODULATE_FORCE) and (mod_cmd.target_force_n == 5.2)
    total_checks += 1
    if print_check(
        "Holding State Grip Modulation",
        holding_mod_ok,
        f"Command: {mod_cmd.action} | Force: {mod_cmd.target_force_n:.2f}N | Grip Maintained",
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 2: Voluntary Extensor Muscle Signal Conditioning & HAND_OPEN Decoding
    # --------------------------------------------------------------------------
    print_header("2. Extensor Muscle Signal Conditioning & HAND_OPEN Decoding")
    fs = float(EMG_SAMPLING_RATE_HZ)
    emg_decoder = EMGDecoder(num_channels=2, fs=fs)

    # Ch 0: Flexor resting (2 uV), Ch 1: Extensor voluntary burst (140 uV @ 100 Hz)
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    extensor_burst = np.vstack([
        2.0 * np.random.randn(len(t)),
        140.0 * np.sin(2 * np.pi * 100.0 * t),
    ])
    extensor_intent = emg_decoder.update_samples(extensor_burst)

    extensor_decoded_ok = (
        (extensor_intent.gesture == EMGIntent.HAND_OPEN)
        and (extensor_intent.activation_level > 0.50)
        and (extensor_intent.confidence > 0.85)
    )
    total_checks += 1
    if print_check(
        "Extensor Intent Decoding (HAND_OPEN)",
        extensor_decoded_ok,
        f"Gesture: {extensor_intent.gesture} | Act: {extensor_intent.activation_level*100:.1f}% | Conf: {extensor_intent.confidence:.2f}",
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 3: Multimodal Intent Fusion Release Arbitration & Neural Disturbance Rejection
    # --------------------------------------------------------------------------
    print_header("3. Multimodal Intent Fusion Release Arbitration & Disturbance Rejection")
    # A. Voluntary release command from extensor
    release_cmd = fusion.fuse(
        emg_intent=extensor_intent,
        eeg_intent=EEGIntent(EEGIntent.IDLE, 0.5, 0.2, 0.1, 0.4, False, time.time()),
        visible_targets=[target_mug],
        system_state="HOLDING",
    )
    voluntary_release_ok = (release_cmd.action == MultimodalCommand.RELEASE_GRIP) and (release_cmd.source == "EMG_RELEASE")

    # B. Neural Disturbance Rejection: Spurious EEG blink in HOLDING with resting EMG must NOT trigger release
    resting_emg = EMGIntent(EMGIntent.REST, 0.95, 0.3, 0.05, [5.0, 5.0], time.time())
    spurious_eeg = EEGIntent(EEGIntent.TARGET_CYCLE_NEXT, 0.90, 0.85, 0.05, 0.2, False, time.time())
    spurious_cmd = fusion.fuse(
        emg_intent=resting_emg,
        eeg_intent=spurious_eeg,
        visible_targets=[target_mug],
        system_state="HOLDING",
    )
    noise_rejected_ok = (spurious_cmd.action == MultimodalCommand.NO_OP)

    total_checks += 1
    if print_check(
        "Fusion Release Arbitration & Neural Immunity",
        voluntary_release_ok and noise_rejected_ok,
        f"Release Action: {release_cmd.action} (Source: {release_cmd.source}) | Spurious EEG Rejected: {noise_rejected_ok}",
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 4: Synchronized 5-Finger Hand Extension (PCA9685 CH0 - CH4)
    # --------------------------------------------------------------------------
    print_header("4. Synchronized 5-Finger Extension Actuation (CH0 - CH4)")
    arm = ArmController(force_mock=force_mock)

    # Setup arm in grasped state (fingers closed onto object at 3.5N)
    arm.set_gripper_opening_percent(35.0)
    if isinstance(arm.driver, MockServoDriver):
        arm.driver.set_tactile_force(3.5)
        arm.current_force_n = 3.5

    # Execute release
    arm.release_grasp(open_percent=100.0)
    angles_after_release = arm.get_joint_angles()

    # Verify all 5 finger angles opened to 0.0°
    fingers = ["finger_thumb", "finger_index", "finger_middle", "finger_ring", "finger_pinky"]
    all_fingers_open = all(abs(angles_after_release.get(f, -1.0) - 0.0) < 0.1 for f in fingers)
    tactile_cleared = (arm.current_force_n == 0.0)

    total_checks += 1
    if print_check(
        "5-Finger Full Extension & Force Clearance",
        all_fingers_open and tactile_cleared,
        f"All 5 fingers at 0.0° flexion (100% open aperture) | Tactile Force: {arm.current_force_n:.2f}N",
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 5: Clean Standoff Clearance Disengagement Kinematics
    # --------------------------------------------------------------------------
    print_header("5. Clean Standoff Clearance Disengagement Kinematics")
    arm_model = ArmModel()
    fk = ForwardKinematics(arm_model)
    ik = InverseKinematics(arm_model)

    # Object contact position
    contact_pos = target_mug.center_base_m
    contact_q, c_ok = ik.solve_analytical(contact_pos, target_pitch_deg=0.0)
    arm.move_to_angles(contact_q, duration_s=0.1)

    # Disengagement standoff: move back 4cm along approach vector
    standoff_clearance_pos = target_mug.center_base_m - 0.04 * target_mug.approach_vector
    standoff_q, s_ok = ik.solve_analytical(standoff_clearance_pos, target_pitch_deg=0.0)
    
    # Verify IK convergence on standoff
    fk_standoff = fk.compute_fk(standoff_q)["ee_position"]
    clearance_dist_err = np.linalg.norm(standoff_clearance_pos - fk_standoff)
    disengage_ik_ok = s_ok and (clearance_dist_err < 0.015)

    total_checks += 1
    if print_check(
        "Disengagement Standoff IK Resolution",
        disengage_ik_ok,
        f"Clearance Standoff: {standoff_clearance_pos.round(3)}m | IK Error: {clearance_dist_err*1000:.2f} mm",
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 6: Quintic Polynomial Minimum-Jerk Retraction Trajectory Planning
    # --------------------------------------------------------------------------
    print_header("6. Quintic Minimum-Jerk Retraction Trajectory Planning")
    planner = TrajectoryPlanner()

    # Plan trajectory from standoff clearance pose back to home pose
    home_angles = arm_model.home_angles_deg
    retract_traj = planner.plan_trajectory(standoff_q, home_angles, min_duration_s=1.2)

    # Verify trajectory validity
    has_waypoints = len(retract_traj) >= 20
    t0_pt = retract_traj[0]
    t_end_pt = retract_traj[-1]
    
    # Boundary conditions: zero velocity at start and end
    v0_max = max(abs(v) for v in t0_pt.velocities_deg_s.values())
    vT_max = max(abs(v) for v in t_end_pt.velocities_deg_s.values())
    boundary_vel_ok = (v0_max < 1e-4) and (vT_max < 1e-4)

    total_checks += 1
    if print_check(
        "Quintic Retraction Trajectory Profile",
        has_waypoints and boundary_vel_ok,
        f"Waypoints: {len(retract_traj)} | Duration: {t_end_pt.time_s:.2f}s | Boundary Vel: v0={v0_max:.4f}, vT={vT_max:.4f}°/s",
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 7: Multi-Servo Retraction Trajectory Execution & Home Convergence
    # --------------------------------------------------------------------------
    print_header("7. Multi-Servo Retraction Trajectory Execution & Home Convergence")
    # Move arm to standoff pose first
    arm.move_to_angles(standoff_q, duration_s=0.2)
    
    # Execute smooth retraction trajectory
    exec_ok = arm.execute_trajectory(retract_traj, abort_check_fn=lambda: arm.is_emergency_stopped)
    
    # Verify convergence to home pose
    final_angles = arm.get_joint_angles()
    arm_joints = ["joint_1_base_yaw", "joint_2_shoulder_pitch", "joint_3_elbow_pitch", "joint_4_wrist_pitch", "joint_5_wrist_roll"]
    max_home_err = max(abs(final_angles.get(j, 0.0) - home_angles[j]) for j in arm_joints)
    home_converged_ok = exec_ok and (max_home_err < 0.5)

    total_checks += 1
    if print_check(
        "Arm Safe Retraction to Home Pose",
        home_converged_ok,
        f"Execution: Success | Max Joint Error to Home: {max_home_err:.3f}° (< 0.5° tolerance)",
    ):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 8: FSM Lifecycle Completion (HOLDING -> RELEASING -> RETRACTING -> IDLE)
    # --------------------------------------------------------------------------
    print_header("8. Finite State Machine Complete Lifecycle Return to IDLE")
    sm = VAPAStateMachine(initial_state=VAPAState.HOLDING)
    sm.transition_to(VAPAState.RELEASING, target=target_mug)
    releasing_state_ok = (sm.current_state == VAPAState.RELEASING)

    sm.transition_to(VAPAState.RETRACTING, target=target_mug)
    retracting_state_ok = (sm.current_state == VAPAState.RETRACTING)

    # Return to IDLE and clear target
    sm.transition_to(VAPAState.IDLE)
    sm.clear_target()
    idle_reset_ok = (sm.current_state == VAPAState.IDLE) and (sm.locked_target is None)

    fsm_cycle_ok = releasing_state_ok and retracting_state_ok and idle_reset_ok
    total_checks += 1
    if print_check(
        "FSM Full Lifecycle Cycle to IDLE",
        fsm_cycle_ok,
        f"Sequence: HOLDING -> RELEASING -> RETRACTING -> IDLE | Locked Target: None (Reset)",
    ):
        passed_checks += 1

    # Cleanup
    arm.release_grasp()
    arm.go_to_home(duration_s=0.2)
    arm.close()

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    elapsed_s = time.time() - start_time
    print("\n" + "=" * 78)
    print(f" {Color.BOLD}PHASE 4 RELEASE & RETRACT DIAGNOSTICS SUMMARY{Color.RESET}")
    print("=" * 78)
    print(f" Total Phase 4 Checks      : {total_checks}")
    print(f" Passed                    : {Color.GREEN}{passed_checks}{Color.RESET}")
    print(f" Failed                    : {Color.RED}{total_checks - passed_checks}{Color.RESET}")
    print(f" Execution Duration        : {elapsed_s:.3f} seconds")
    print(f" Final System State        : {Color.GREEN}IDLE (Ready for Next Perception & Grasp Cycle){Color.RESET}")

    if passed_checks == total_checks:
        print(f"\n {Color.BOLD}{Color.GREEN}>>> ALL PHASE 4 CHECKS PASSED — ARM SAFELY RETRACTED TO HOME! <<<{Color.RESET}\n")
        return True
    else:
        print(f"\n {Color.BOLD}{Color.RED}>>> PHASE 4 ISSUES DETECTED <<<{Color.RESET}\n")
        return False


if __name__ == "__main__":
    force_mock_flag = "--mock" in sys.argv or "--real" not in sys.argv
    success = run_phase_4_release_retract(force_mock=force_mock_flag)
    sys.exit(0 if success else 1)
