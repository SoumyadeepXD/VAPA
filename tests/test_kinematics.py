"""
VAPA Subsystem Test 3: Kinematics (FK, IK, Workspace & Trajectory Planning)
Validates Forward Kinematics, Analytical/Numerical IK convergence,
and minimum-jerk trajectory generation.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinematics.arm_model import ArmModel
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import InverseKinematics
from kinematics.trajectory_planner import TrajectoryPlanner


def test_kinematics():
    print("=" * 75)
    print(" VAPA TEST: Robotic Arm Kinematics & Motion Planning")
    print("=" * 75)

    model = ArmModel()
    fk = ForwardKinematics(model)
    ik = InverseKinematics(model)
    planner = TrajectoryPlanner()

    print(f"\nArm Model Link Lengths: L1={model.L1*100:.1f}cm, L2={model.L2*100:.1f}cm, L3={model.L3*100:.1f}cm, L4={model.L4*100:.1f}cm, L5={model.L5*100:.1f}cm")
    print(f"Max Reach: {model.max_reach*100:.1f} cm, Min Reach: {model.min_reach*100:.1f} cm")

    # 1. Forward Kinematics on Home Pose
    print("\n--- 1. Forward Kinematics Test ---")
    home_fk = fk.compute_fk(model.home_angles_deg)
    p_tcp = home_fk["ee_position"]
    rpy = home_fk["ee_orientation_rpy"]
    print(f"Home Joint Angles: {model.home_angles_deg}")
    print(f"Home TCP Position: [X: {p_tcp[0]:.3f}, Y: {p_tcp[1]:.3f}, Z: {p_tcp[2]:.3f}] meters")
    print(f"Home Orientation: [Roll: {rpy[0]:.1f}, Pitch: {rpy[1]:.1f}, Yaw: {rpy[2]:.1f}] degrees")

    # 2. Inverse Kinematics Reachability Grid Test
    print("\n--- 2. Inverse Kinematics Grid Validation ---")
    test_points = [
        np.array([0.25, 0.00, 0.15]),   # Straight ahead, mid height
        np.array([0.20, 0.15, 0.10]),   # Forward-left, lower
        np.array([0.20, -0.15, 0.10]),  # Forward-right, lower
        np.array([0.32, 0.00, 0.05]),   # Extended reach tabletop
        np.array([0.18, 0.08, 0.25]),   # Close reach high
    ]

    passed = 0
    for idx, target in enumerate(test_points):
        angles, success = ik.solve_analytical(target, target_pitch_deg=0.0)
        # Compute FK of solved angles to measure residual position error
        fk_result = fk.compute_fk(angles)
        actual_pos = fk_result["ee_position"]
        error_mm = np.linalg.norm(target - actual_pos) * 1000.0

        status = "PASSED" if (success and error_mm < 15.0) else "FAILED"
        if status == "PASSED":
            passed += 1

        print(
            f"  Point #{idx+1}: Target=[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]m -> "
            f"Actual=[{actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f}]m | "
            f"Err={error_mm:4.1f}mm | {status}"
        )

    print(f"\nIK Verification Result: {passed}/{len(test_points)} test targets reached with < 15mm error.")

    # 3. Trajectory Planner Test
    print("\n--- 3. Trajectory Planner Smooth Interpolation ---")
    start_q = model.home_angles_deg.copy()
    target_q, _ = ik.solve_analytical(test_points[0])
    trajectory = planner.plan_trajectory(start_q, target_q, min_duration_s=1.0)

    print(f"Generated trajectory with {len(trajectory)} waypoints over {trajectory[-1].time_s:.2f} seconds.")
    print("Waypoint samples:")
    for i in [0, len(trajectory) // 4, len(trajectory) // 2, 3 * len(trajectory) // 4, len(trajectory) - 1]:
        pt = trajectory[i]
        j1 = pt.angles_deg.get("joint_1_base_yaw", 0.0)
        j2 = pt.angles_deg.get("joint_2_shoulder_pitch", 0.0)
        j3 = pt.angles_deg.get("joint_3_elbow_pitch", 0.0)
        print(f"  t={pt.time_s:4.2f}s | J1={j1:5.1f}deg, J2={j2:5.1f}deg, J3={j3:5.1f}deg")

    print("\nKinematics Verification Complete!")


if __name__ == "__main__":
    test_kinematics()
