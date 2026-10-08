"""
VAPA Mock Simulation Servo Driver
Simulates multi-servo hardware with joint angle tracking, motor current draw,
and tactile contact force response for offline development and testing.
"""

import time
import logging
import numpy as np

from actuation.servo_interface import BaseServoDriver
from config.hardware_config import SERVO_CHANNELS
from config.system_config import JOINT_LIMITS_DEG

logger = logging.getLogger("VAPA.Actuation.Mock")


class MockServoDriver(BaseServoDriver):
    """Virtual simulation driver for testing without physical arm hardware."""
    def __init__(self):
        self.target_angles = {k: JOINT_LIMITS_DEG[k][2] for k in JOINT_LIMITS_DEG}
        for ch_name, cfg in SERVO_CHANNELS.items():
            if ch_name not in self.target_angles:
                self.target_angles[ch_name] = cfg.get("home_deg", 0.0)
        self.current_angles = self.target_angles.copy()
        self.current_pulses = {k: 1500.0 for k in SERVO_CHANNELS}
        self.applied_tactile_force_n = 0.0
        self.is_holding_object = False
        self.object_stiffness = 50.0  # N/m
        logger.info("MockServoDriver initialized in software simulation mode.")

    def set_servo_pulse(self, channel: int, pulse_us: float):
        for name, cfg in SERVO_CHANNELS.items():
            if cfg["channel"] == channel:
                self.current_pulses[name] = float(pulse_us)
                # Map pulse to angle
                ratio = (pulse_us - cfg["min_us"]) / max(1e-6, (cfg["max_us"] - cfg["min_us"]))
                angle = cfg["min_angle_deg"] + ratio * (cfg["max_angle_deg"] - cfg["min_angle_deg"])
                self.current_angles[name] = float(angle)
                break

    def set_joint_angle(self, joint_name: str, angle_deg: float):
        self.target_angles[joint_name] = float(angle_deg)
        if joint_name not in self.current_angles:
            self.current_angles[joint_name] = float(angle_deg)
        else:
            # Smoothly update current angle toward target
            diff = angle_deg - self.current_angles[joint_name]
            step = np.clip(diff, -15.0, 15.0)
            self.current_angles[joint_name] += step

    def set_all_angles(self, angles_deg: dict[str, float]):
        for j_name, angle in angles_deg.items():
            self.set_joint_angle(j_name, angle)

    def set_tactile_force(self, force_n: float):
        self.applied_tactile_force_n = float(force_n)

    def read_feedback(self) -> dict:
        # Simulate current draw proportional to gripper closing and arm load
        gripper_closed_ratio = 1.0 - (self.current_angles.get("joint_6_gripper", 100.0) / 100.0)
        simulated_current_a = 0.15 + 0.8 * gripper_closed_ratio + 0.1 * np.random.normal(0, 0.02)
        
        return {
            "angles_deg": self.current_angles.copy(),
            "target_angles_deg": self.target_angles.copy(),
            "current_a": max(0.05, float(simulated_current_a)),
            "tactile_n": float(self.applied_tactile_force_n),
            "temperature_c": 32.5,
            "bus_voltage_v": 5.15,
            "connected": True,
        }

    def emergency_stop(self):
        logger.warning("MOCK SERVO EMERGENCY STOP: Holding current joint angles.")
        self.target_angles = self.current_angles.copy()

    def close(self):
        logger.info("MockServoDriver closed.")
