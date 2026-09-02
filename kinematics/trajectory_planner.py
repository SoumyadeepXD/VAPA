"""
VAPA Trajectory Planner & Smooth Motion Interpolator
Generates minimum-jerk and quintic polynomial trajectories between joint configurations,
ensuring smooth, vibration-free prosthetic arm movement within safe velocity limits.
"""

import math
import numpy as np
from config.system_config import (
    MAX_JOINT_VELOCITY_DEG_S,
    MAX_JOINT_ACCELERATION_DEG_S2,
    TRAJECTORY_INTERPOLATION_HZ,
)


class JointTrajectoryPoint:
    """Represents a single timestamped waypoint in joint space."""
    def __init__(self, time_s: float, angles_deg: dict[str, float], velocities_deg_s: dict[str, float] = None):
        self.time_s = float(time_s)
        self.angles_deg = angles_deg
        self.velocities_deg_s = velocities_deg_s or {k: 0.0 for k in angles_deg}

    def __repr__(self):
        return f"JointTrajectoryPoint(t={self.time_s:.2f}s, angles={self.angles_deg})"


class TrajectoryPlanner:
    """Plans smooth multi-joint trajectories."""
    def __init__(
        self,
        max_velocity_deg_s: float = MAX_JOINT_VELOCITY_DEG_S,
        max_acceleration_deg_s2: float = MAX_JOINT_ACCELERATION_DEG_S2,
        rate_hz: int = TRAJECTORY_INTERPOLATION_HZ,
    ):
        self.max_v = float(max_velocity_deg_s)
        self.max_a = float(max_acceleration_deg_s2)
        self.rate_hz = int(rate_hz)
        self.dt = 1.0 / self.rate_hz

    def plan_trajectory(
        self, start_angles: dict[str, float], target_angles: dict[str, float], min_duration_s: float = 0.5
    ) -> list[JointTrajectoryPoint]:
        """
        Computes minimum-jerk trajectory waypoints from start to target joint angles.
        Interpolation profile: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 (tau in [0, 1])
        """
        # Determine duration based on maximum joint displacement and max velocity
        max_displacement = 0.0
        all_joints = list(set(start_angles.keys()).union(target_angles.keys()))

        for j in all_joints:
            q0 = start_angles.get(j, 0.0)
            q1 = target_angles.get(j, 0.0)
            disp = abs(q1 - q0)
            if disp > max_displacement:
                max_displacement = disp

        # Duration based on max velocity (with factor of 1.5 for acceleration ramp)
        duration_s = max(min_duration_s, (max_displacement / self.max_v) * 1.5)
        num_steps = max(2, int(math.ceil(duration_s * self.rate_hz)))

        trajectory = []
        for step in range(num_steps + 1):
            t = step * self.dt
            tau = min(1.0, t / duration_s)

            # Minimum Jerk polynomial (zero velocity and zero acceleration at t=0 and t=T)
            s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            s_dot = (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / duration_s

            step_angles = {}
            step_vels = {}
            for j in all_joints:
                q0 = start_angles.get(j, 0.0)
                q1 = target_angles.get(j, 0.0)
                step_angles[j] = float(q0 + (q1 - q0) * s)
                step_vels[j] = float((q1 - q0) * s_dot)

            trajectory.append(JointTrajectoryPoint(time_s=t, angles_deg=step_angles, velocities_deg_s=step_vels))

        return trajectory
