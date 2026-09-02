"""
VAPA Servo Controller Abstract Interface
Defines common interface for I2C (PCA9685), Serial Bus, and Mock servo drivers.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseServoDriver(ABC):
    """Abstract interface for robotic arm multi-servo drivers."""

    @abstractmethod
    def set_servo_pulse(self, channel: int, pulse_us: float):
        """Sets raw PWM pulse width in microseconds on a specific channel."""
        pass

    @abstractmethod
    def set_joint_angle(self, joint_name: str, angle_deg: float):
        """Sets calibrated joint angle in degrees for a named joint."""
        pass

    @abstractmethod
    def set_all_angles(self, angles_deg: dict[str, float]):
        """Sets all joint angles simultaneously."""
        pass

    @abstractmethod
    def read_feedback(self) -> dict:
        """Reads motor currents, temperatures, tactile force, and current angles."""
        pass

    @abstractmethod
    def emergency_stop(self):
        """Immediately disables PWM signals or holds current position."""
        pass

    @abstractmethod
    def close(self):
        """Releases hardware resources."""
        pass
