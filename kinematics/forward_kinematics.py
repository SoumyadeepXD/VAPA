"""
VAPA 3D Forward Kinematics (FK)
Computes Cartesian positions of all arm joints and end-effector pose from joint angles.
"""

import math
import numpy as np
from kinematics.arm_model import ArmModel


class ForwardKinematics:
    """Calculates forward kinematics and 3D joint skeleton positions."""
    def __init__(self, model: ArmModel = None):
        self.model = model or ArmModel()

    def compute_fk(self, angles_deg: dict[str, float]) -> dict:
        """
        Calculates end-effector Cartesian coordinates and intermediate joint positions.
        angles_deg: dictionary containing joint angles in degrees.
        Returns:
          {
            "ee_position": np.ndarray [x, y, z] in meters,
            "ee_orientation_rpy": np.ndarray [roll, pitch, yaw] in degrees,
            "joint_positions": list of np.ndarray [x, y, z] for base, shoulder, elbow, wrist, palm, tcp,
          }
        """
        # Extract angles in radians
        q1 = math.radians(angles_deg.get("joint_1_base_yaw", 0.0))
        q2 = math.radians(angles_deg.get("joint_2_shoulder_pitch", 0.0))
        q3 = math.radians(angles_deg.get("joint_3_elbow_pitch", 0.0))
        q4 = math.radians(angles_deg.get("joint_4_wrist_pitch", 0.0))
        q5 = math.radians(angles_deg.get("joint_5_wrist_roll", 0.0))

        L1 = self.model.L1  # Base height
        L2 = self.model.L2  # Upper arm
        L3 = self.model.L3  # Forearm
        L4 = self.model.L4  # Wrist to palm
        L5 = self.model.L5  # Finger reach

        # 1. Base Joint Origin
        p_base = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 2. Shoulder Joint Position (Elevated by L1 along Z)
        p_shoulder = np.array([0.0, 0.0, L1], dtype=np.float32)

        # 2D Arm Plane Kinematics (in the vertical plane rotated by base yaw q1)
        # Angle of upper arm relative to horizontal: alpha2 = (90 - q2) or q2
        # Let q2 = 0 be horizontal pointing forward, q2 = 90 pointing up
        # Cumulative pitch angles in the plane:
        theta_upper = q2
        theta_fore = q2 + q3
        theta_wrist = q2 + q3 + q4

        # In-plane coordinates (r = radial distance along ground, z = height above shoulder)
        r_elbow = L2 * math.cos(theta_upper)
        z_elbow = L1 + L2 * math.sin(theta_upper)
        p_elbow = np.array([r_elbow * math.cos(q1), r_elbow * math.sin(q1), z_elbow], dtype=np.float32)

        r_wrist = r_elbow + L3 * math.cos(theta_fore)
        z_wrist = z_elbow + L3 * math.sin(theta_fore)
        p_wrist = np.array([r_wrist * math.cos(q1), r_wrist * math.sin(q1), z_wrist], dtype=np.float32)

        r_palm = r_wrist + L4 * math.cos(theta_wrist)
        z_palm = z_wrist + L4 * math.sin(theta_wrist)
        p_palm = np.array([r_palm * math.cos(q1), r_palm * math.sin(q1), z_palm], dtype=np.float32)

        # Tool Center Point (TCP / Finger tips)
        r_tcp = r_palm + L5 * math.cos(theta_wrist)
        z_tcp = z_palm + L5 * math.sin(theta_wrist)
        p_tcp = np.array([r_tcp * math.cos(q1), r_tcp * math.sin(q1), z_tcp], dtype=np.float32)

        # Orientation
        pitch_deg = math.degrees(theta_wrist)
        yaw_deg = math.degrees(q1)
        roll_deg = math.degrees(q5)

        return {
            "ee_position": p_tcp,
            "ee_orientation_rpy": np.array([roll_deg, pitch_deg, yaw_deg], dtype=np.float32),
            "joint_positions": [p_base, p_shoulder, p_elbow, p_wrist, p_palm, p_tcp],
        }
