"""
VAPA Inverse Kinematics (IK) Solver
Solves target 3D Cartesian points into joint angles using Analytical Closed-Form
and Numerical Damped Least Squares (DLS) methods with joint limit constraints.
"""

import math
import logging
import numpy as np
from kinematics.arm_model import ArmModel
from kinematics.forward_kinematics import ForwardKinematics

logger = logging.getLogger("VAPA.Kinematics.IK")


class InverseKinematics:
    """Computes joint angles for given Cartesian 3D target coordinates."""
    def __init__(self, model: ArmModel = None):
        self.model = model or ArmModel()
        self.fk = ForwardKinematics(self.model)

    def solve_analytical(
        self,
        target_pos_m: np.ndarray,
        target_pitch_deg: float = 0.0,
        target_roll_deg: float = 0.0,
        gripper_percent: float = 100.0,
    ) -> tuple[dict[str, float], bool]:
        """
        Solves closed-form geometric IK for 6-DOF robotic arm.
        target_pos_m: [x, y, z] target coordinates in robot base frame (meters).
        Returns: (joint_angles_deg_dict, success_bool)
        """
        x, y, z = float(target_pos_m[0]), float(target_pos_m[1]), float(target_pos_m[2])

        # 1. Solve Joint 1 (Base Yaw)
        r_total = math.sqrt(x**2 + y**2)
        if r_total < 1e-4:
            q1_deg = 0.0
        else:
            q1_deg = math.degrees(math.atan2(y, x))

        # Check Joint 1 bounds
        lim1 = self.model.limits["joint_1_base_yaw"]
        if not (lim1[0] <= q1_deg <= lim1[1]):
            logger.warning(f"Target yaw {q1_deg:.1f}deg out of joint 1 limits [{lim1[0]}, {lim1[1]}]")
            return self.model.home_angles_deg.copy(), False

        # Link lengths
        L1 = self.model.L1
        L2 = self.model.L2
        L3 = self.model.L3
        L_wrist_to_tcp = self.model.L4 + self.model.L5

        # 2. Find Wrist Position (step back from TCP along target approach pitch)
        pitch_rad = math.radians(target_pitch_deg)
        r_wrist = r_total - L_wrist_to_tcp * math.cos(pitch_rad)
        z_wrist = z - L1 - L_wrist_to_tcp * math.sin(pitch_rad)

        # Distance from shoulder joint to wrist center
        d_sq = r_wrist**2 + z_wrist**2
        d = math.sqrt(d_sq)

        # Check reachability to wrist
        if d > (L2 + L3) or d < abs(L2 - L3):
            logger.warning(f"Target position [{x:.3f}, {y:.3f}, {z:.3f}] unreachable (wrist dist {d:.3f}m > {L2+L3:.3f}m)")
            return self.model.home_angles_deg.copy(), False

        # 3. Solve Joint 3 (Elbow Pitch) using Law of Cosines
        cos_q3 = (d_sq - L2**2 - L3**2) / (2.0 * L2 * L3)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        # Elbow-up branch (negative angle)
        q3_rad = -math.acos(cos_q3)
        q3_deg = math.degrees(q3_rad)

        # 4. Solve Joint 2 (Shoulder Pitch)
        alpha = math.atan2(z_wrist, r_wrist)
        beta = math.atan2(L3 * math.sin(-q3_rad), L2 + L3 * math.cos(q3_rad))
        q2_rad = alpha + beta
        q2_deg = math.degrees(q2_rad)

        # 5. Solve Joint 4 (Wrist Pitch) to match target orientation
        # Total pitch = q2 + q3 + q4 => q4 = target_pitch - q2 - q3
        q4_deg = target_pitch_deg - (q2_deg + q3_deg)

        # 6. Joint 5 (Wrist Roll) & Joint 6 (Gripper)
        q5_deg = target_roll_deg
        q6_deg = float(gripper_percent)

        angles = {
            "joint_1_base_yaw": q1_deg,
            "joint_2_shoulder_pitch": q2_deg,
            "joint_3_elbow_pitch": q3_deg,
            "joint_4_wrist_pitch": q4_deg,
            "joint_5_wrist_roll": q5_deg,
            "joint_6_gripper": q6_deg,
        }

        # Check joint limits
        clamped = self.model.clamp_angles(angles)
        is_exact = all(abs(clamped[k] - angles[k]) < 0.1 for k in angles)

        # Verify forward kinematics residual
        fk_res = self.fk.compute_fk(clamped)
        pos_err = np.linalg.norm(fk_res["ee_position"] - np.array([x, y, z]))

        if not is_exact or pos_err > 0.035:
            logger.debug(f"Analytical IK clamped or pos error {pos_err*100:.1f}cm. Trying numerical refinement.")
            return self.solve_numerical_dls(target_pos_m, initial_angles=clamped)

        return clamped, True

    def solve_numerical_dls(
        self,
        target_pos_m: np.ndarray,
        initial_angles: dict[str, float] = None,
        max_iters: int = 50,
        tolerance_m: float = 0.005,
        damping_lambda: float = 0.05,
    ) -> tuple[dict[str, float], bool]:
        """
        Damped Least Squares (DLS) Jacobian numerical IK solver.
        Refines angles iteratively to reach target position with minimal residual error.
        """
        current_angles = (initial_angles or self.model.home_angles_deg).copy()
        target = np.asarray(target_pos_m, dtype=np.float32)

        active_joints = ["joint_1_base_yaw", "joint_2_shoulder_pitch", "joint_3_elbow_pitch", "joint_4_wrist_pitch"]
        q_vec = np.array([current_angles[j] for j in active_joints], dtype=np.float32)

        for _ in range(max_iters):
            # Compute current FK
            for idx, j_name in enumerate(active_joints):
                current_angles[j_name] = float(q_vec[idx])

            fk_data = self.fk.compute_fk(current_angles)
            cur_pos = fk_data["ee_position"]
            err = target - cur_pos

            if np.linalg.norm(err) < tolerance_m:
                return self.model.clamp_angles(current_angles), True

            # Approximate Jacobian via finite differences
            J = np.zeros((3, len(active_joints)), dtype=np.float32)
            eps_deg = 0.1
            for j_idx, j_name in enumerate(active_joints):
                test_angles = current_angles.copy()
                test_angles[j_name] += eps_deg
                fk_pert = self.fk.compute_fk(test_angles)
                J[:, j_idx] = (fk_pert["ee_position"] - cur_pos) / math.radians(eps_deg)

            # DLS Inversion: delta_q = J^T * (J * J^T + lambda^2 * I)^-1 * error
            JJt = J @ J.T
            damped_inv = np.linalg.inv(JJt + (damping_lambda**2) * np.eye(3))
            delta_q_rad = J.T @ damped_inv @ err
            delta_q_deg = np.degrees(delta_q_rad)

            # Step and clamp
            q_vec += np.clip(delta_q_deg, -10.0, 10.0)
            for idx, j_name in enumerate(active_joints):
                lim = self.model.limits[j_name]
                q_vec[idx] = np.clip(q_vec[idx], lim[0], lim[1])

        # Final evaluation
        for idx, j_name in enumerate(active_joints):
            current_angles[j_name] = float(q_vec[idx])
        fk_final = self.fk.compute_fk(current_angles)
        final_err = np.linalg.norm(target - fk_final["ee_position"])
        success = bool(final_err < 0.020)

        return self.model.clamp_angles(current_angles), success
