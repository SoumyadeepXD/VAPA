"""
VAPA Robotic Arm Kinematic Model
Defines link lengths, DH parameters, joint angle bounds, and default poses.
"""

import numpy as np
from config.system_config import LINK_LENGTHS, JOINT_LIMITS_DEG, WORKSPACE_BOUNDS_M


class ArmModel:
    """Represents physical dimensions, joint names, and kinematic constraints."""
    JOINT_NAMES = [
        "joint_1_base_yaw",
        "joint_2_shoulder_pitch",
        "joint_3_elbow_pitch",
        "joint_4_wrist_pitch",
        "joint_5_wrist_roll",
        "joint_6_gripper",
    ]

    def __init__(self):
        # Link dimensions in meters
        self.L1 = float(LINK_LENGTHS["base_height"])      # Base to Shoulder (0.080m)
        self.L2 = float(LINK_LENGTHS["upper_arm"])        # Shoulder to Elbow (0.180m)
        self.L3 = float(LINK_LENGTHS["forearm"])          # Elbow to Wrist Pitch (0.160m)
        self.L4 = float(LINK_LENGTHS["wrist_to_palm"])    # Wrist to Palm Center (0.085m)
        self.L5 = float(LINK_LENGTHS["finger_length"])    # Finger tip offset (0.065m)

        # Total reach
        self.max_reach = self.L2 + self.L3 + self.L4 + self.L5
        self.min_reach = abs(self.L2 - self.L3) + 0.05

        # Joint Limits in Degrees [Min, Max, Home]
        self.limits = JOINT_LIMITS_DEG
        self.workspace = WORKSPACE_BOUNDS_M

        # Default Home Angles (degrees)
        self.home_angles_deg = {name: self.limits[name][2] for name in self.JOINT_NAMES}

    def clamp_angles(self, angles_deg: dict[str, float]) -> dict[str, float]:
        """Clamps joint angles within physical safety bounds."""
        clamped = {}
        for name, val in angles_deg.items():
            if name in self.limits:
                min_v, max_v, _ = self.limits[name]
                clamped[name] = float(np.clip(val, min_v, max_v))
            else:
                clamped[name] = float(val)
        return clamped

    def is_in_workspace(self, x: float, y: float, z: float) -> bool:
        """Validates if Cartesian position is inside safe workspace boundaries."""
        wb = self.workspace
        r = np.sqrt(x**2 + y**2)
        if r > self.max_reach or r < self.min_reach:
            return False
        return (wb["x_min"] <= x <= wb["x_max"]) and (wb["y_min"] <= y <= wb["y_max"]) and (wb["z_min"] <= z <= wb["z_max"])
