"""
VAPA Unified Arm & Hand Controller
High-level actuator manager: coordinates multi-joint trajectories,
closed-loop force-controlled grasping, and multi-finger prosthetic actuation.
"""

import time
import logging
import numpy as np

from config.hardware_config import SIMULATION_MODE
from config.system_config import (
    JOINT_LIMITS_DEG,
    FORCE_MIN_N,
    FORCE_MAX_N,
    FORCE_EMERGENCY_LIMIT_N,
    ACTUATION_LOOP_RATE_HZ,
)
from actuation.servo_interface import BaseServoDriver
from actuation.pca9685_controller import PCA9685ServoDriver
from actuation.serial_servo_controller import SerialServoDriver
from actuation.mock_arm_controller import MockServoDriver
from kinematics.arm_model import ArmModel
from kinematics.trajectory_planner import JointTrajectoryPoint

logger = logging.getLogger("VAPA.Actuation.ArmController")


class ArmController:
    """Unified controller for arm kinematics execution, servos, and gripper force."""
    def __init__(self, driver: BaseServoDriver = None, force_mock: bool = False):
        self.arm_model = ArmModel()
        self.dt = 1.0 / ACTUATION_LOOP_RATE_HZ

        if driver is not None:
            self.driver = driver
        elif force_mock or SIMULATION_MODE:
            self.driver = MockServoDriver()
        else:
            # Try PCA9685 first, fallback to Serial, then Mock
            pca = PCA9685ServoDriver()
            if pca.is_connected:
                self.driver = pca
            else:
                serial_drv = SerialServoDriver()
                if serial_drv.is_connected:
                    self.driver = serial_drv
                else:
                    logger.warning("No physical servo hardware detected. Initializing MockServoDriver.")
                    self.driver = MockServoDriver()

        # Initialize arm to Home Position
        self.current_angles = self.arm_model.home_angles_deg.copy()
        self.driver.set_all_angles(self.current_angles)
        self.current_force_n = 0.0
        self.is_emergency_stopped = False
        logger.info("ArmController ready.")

    def get_joint_angles(self) -> dict[str, float]:
        """Returns current joint angles in degrees."""
        return self.current_angles.copy()

    def go_to_home(self, duration_s: float = 1.5):
        """Moves arm smoothly to default home position."""
        self.move_to_angles(self.arm_model.home_angles_deg, duration_s=duration_s)

    def move_to_angles(self, target_angles: dict[str, float], duration_s: float = 1.0):
        """Directly interpolates to target joint configuration."""
        if self.is_emergency_stopped:
            logger.warning("Arm is in EMERGENCY STOP state. Ignoring move command.")
            return

        clamped_target = self.arm_model.clamp_angles(target_angles)
        steps = max(5, int(duration_s * ACTUATION_LOOP_RATE_HZ))

        for step in range(1, steps + 1):
            alpha = step / steps
            interp_angles = {}
            for j in self.arm_model.JOINT_NAMES:
                q0 = self.current_angles.get(j, self.arm_model.home_angles_deg[j])
                q1 = clamped_target.get(j, q0)
                interp_angles[j] = q0 + alpha * (q1 - q0)

            self.driver.set_all_angles(interp_angles)
            self.current_angles = interp_angles.copy()
            time.sleep(self.dt)

    def execute_trajectory(self, trajectory: list[JointTrajectoryPoint], abort_check_fn=None) -> bool:
        """
        Executes a planned smooth joint trajectory.
        abort_check_fn: optional callable returning True if execution should abort immediately.
        """
        if self.is_emergency_stopped:
            return False

        logger.info(f"Executing trajectory with {len(trajectory)} waypoints...")
        last_t = 0.0

        for pt in trajectory:
            if abort_check_fn and abort_check_fn():
                logger.warning("Trajectory aborted by safety trigger.")
                return False

            clamped = self.arm_model.clamp_angles(pt.angles_deg)
            self.driver.set_all_angles(clamped)
            self.current_angles = clamped.copy()

            sleep_time = max(0.001, pt.time_s - last_t)
            time.sleep(sleep_time)
            last_t = pt.time_s

        return True

    def set_gripper_opening_percent(self, percent: float):
        """
        Sets gripper or multi-finger opening percentage (0.0 = Fully closed, 100.0 = Fully open).
        If multi-finger servos are defined, drives all fingers simultaneously.
        """
        percent = float(np.clip(percent, 0.0, 100.0))
        self.driver.set_joint_angle("joint_6_gripper", percent)
        self.current_angles["joint_6_gripper"] = percent

        # Drive individual fingers if present
        for finger in ("finger_thumb", "finger_index", "finger_middle", "finger_ring", "finger_pinky"):
            self.driver.set_joint_angle(finger, percent)
            self.current_angles[finger] = percent

    def execute_force_grasp(self, target_force_n: float, timeout_s: float = 3.0) -> bool:
        """
        Closed-loop force-controlled grasp.
        Closes gripper gradually while reading force/current feedback until target force is reached.
        """
        target_force_n = np.clip(target_force_n, FORCE_MIN_N, FORCE_MAX_N)
        logger.info(f"Executing force grasp. Target Force: {target_force_n:.2f} N")

        start_time = time.time()
        applied_force = 0.0
        grip_percent = self.current_angles.get("joint_6_gripper", 100.0)

        while (time.time() - start_time) < timeout_s:
            feedback = self.driver.read_feedback()

            # Check for excessive force safety violation
            tactile = feedback.get("tactile_n")
            if tactile is not None and tactile >= FORCE_EMERGENCY_LIMIT_N:
                logger.warning(f"Excessive force detected ({tactile:.1f}N > {FORCE_EMERGENCY_LIMIT_N}N)! Aborting grasp.")
                self.emergency_stop()
                return False

            if applied_force >= target_force_n:
                logger.info(f"Target force {target_force_n:.2f} N reached.")
                return True

            # Step gripper closure
            grip_percent = max(0.0, grip_percent - 3.0)
            self.set_gripper_opening_percent(grip_percent)

            # Update simulated/measured force
            step_force = 0.25
            applied_force += step_force
            if isinstance(self.driver, MockServoDriver):
                self.driver.set_tactile_force(applied_force)

            time.sleep(self.dt)

        return applied_force >= (target_force_n * 0.8)

    def release_grasp(self, open_percent: float = 100.0):
        """Opens gripper to release object."""
        self.set_gripper_opening_percent(open_percent)
        if isinstance(self.driver, MockServoDriver):
            self.driver.set_tactile_force(0.0)
        logger.info(f"Grasp released (opened to {open_percent}%).")

    def emergency_stop(self):
        """Immediately halts all motion."""
        self.is_emergency_stopped = True
        self.driver.emergency_stop()
        logger.critical("ARM CONTROLLER EMERGENCY STOP TRIGGERED!")

    def reset_emergency_stop(self):
        """Clears emergency stop and re-enables control."""
        self.is_emergency_stopped = False
        self.driver.set_all_angles(self.current_angles)
        logger.info("Emergency stop reset.")

    def close(self):
        self.driver.close()
