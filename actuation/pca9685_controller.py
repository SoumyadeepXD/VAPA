"""
VAPA PCA9685 I2C 16-Channel 12-bit PWM Servo Driver
Direct I2C communication on NVIDIA Jetson Orin (I2C Bus 1 / Bus 7/8).
Supports calibrated angle-to-PWM translation for multi-servo robotic arms.
"""

import time
import math
import logging
import numpy as np

from actuation.servo_interface import BaseServoDriver
from config.hardware_config import (
    PCA9685_I2C_BUS,
    PCA9685_I2C_ADDRESS,
    PCA9685_PWM_FREQ_HZ,
    SERVO_CHANNELS,
)

logger = logging.getLogger("VAPA.Actuation.PCA9685")

PCA9685_MODE1 = 0x00
PCA9685_PRESCALE = 0xFE
LED0_ON_L = 0x06
LED0_ON_H = 0x07
LED0_OFF_L = 0x08
LED0_OFF_H = 0x09


class PCA9685ServoDriver(BaseServoDriver):
    """Controls multi-servo arm joints via PCA9685 I2C module on Jetson Orin."""
    def __init__(self, bus_num=PCA9685_I2C_BUS, address=PCA9685_I2C_ADDRESS, freq_hz=PCA9685_PWM_FREQ_HZ):
        self.bus_num = bus_num
        self.address = address
        self.freq_hz = freq_hz
        self.i2c_bus = None
        self.current_angles = {}
        self.is_connected = False

        self._init_i2c()

    def _init_i2c(self):
        try:
            try:
                import smbus2 as smbus
            except ImportError:
                import smbus
            self.i2c_bus = smbus.SMBus(self.bus_num)

            # Reset PCA9685 MODE1 register
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE1, 0x00)
            time.sleep(0.01)

            # Set PWM Frequency (50Hz standard for servos)
            prescale_val = int(math.floor(25000000.0 / (4096.0 * self.freq_hz) - 0.5))
            old_mode = self.i2c_bus.read_byte_data(self.address, PCA9685_MODE1)
            new_mode = (old_mode & 0x7F) | 0x10  # Sleep mode to set prescaler
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE1, new_mode)
            self.i2c_bus.write_byte_data(self.address, PCA9685_PRESCALE, prescale_val)
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE1, old_mode)
            time.sleep(0.005)
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE1, old_mode | 0xA1)

            self.is_connected = True
            logger.info(f"PCA9685 initialized on I2C bus {self.bus_num} at address 0x{self.address:02X} @ {self.freq_hz}Hz.")
        except Exception as e:
            logger.warning(f"Could not connect to PCA9685 I2C on bus {self.bus_num}: {e}")
            self.is_connected = False

    def set_pwm(self, channel: int, on_tick: int, off_tick: int):
        if not self.is_connected or self.i2c_bus is None:
            return
        try:
            reg_base = LED0_ON_L + 4 * channel
            self.i2c_bus.write_byte_data(self.address, reg_base, on_tick & 0xFF)
            self.i2c_bus.write_byte_data(self.address, reg_base + 1, (on_tick >> 8) & 0xFF)
            self.i2c_bus.write_byte_data(self.address, reg_base + 2, off_tick & 0xFF)
            self.i2c_bus.write_byte_data(self.address, reg_base + 3, (off_tick >> 8) & 0xFF)
        except Exception as e:
            logger.error(f"Error setting PWM on channel {channel}: {e}")

    def set_servo_pulse(self, channel: int, pulse_us: float):
        period_us = 1000000.0 / self.freq_hz
        pulse_us = max(400.0, min(2600.0, float(pulse_us)))
        off_tick = int(round((pulse_us / period_us) * 4096.0))
        off_tick = max(0, min(4095, off_tick))
        self.set_pwm(channel, 0, off_tick)

    def angle_to_pulse_us(self, joint_name: str, angle_deg: float) -> float:
        cfg = SERVO_CHANNELS.get(joint_name)
        if not cfg:
            return 1500.0

        min_a = cfg["min_angle_deg"]
        max_a = cfg["max_angle_deg"]
        min_u = cfg["min_us"]
        max_u = cfg["max_us"]
        offset = cfg.get("center_offset_deg", 0.0)
        direction = cfg.get("direction", 1)

        adj_angle = (angle_deg + offset) * direction
        clamped_angle = np.clip(adj_angle, min_a, max_a)

        ratio = (clamped_angle - min_a) / max(1e-6, (max_a - min_a))
        pulse_us = min_u + ratio * (max_u - min_u)
        return float(pulse_us)

    def set_joint_angle(self, joint_name: str, angle_deg: float):
        if joint_name not in SERVO_CHANNELS:
            logger.warning(f"Unknown joint: {joint_name}")
            return

        cfg = SERVO_CHANNELS[joint_name]
        ch = cfg["channel"]
        pulse = self.angle_to_pulse_us(joint_name, angle_deg)
        self.set_servo_pulse(ch, pulse)
        self.current_angles[joint_name] = float(angle_deg)

    def set_all_angles(self, angles_deg: dict[str, float]):
        for j_name, angle in angles_deg.items():
            self.set_joint_angle(j_name, angle)

    def read_feedback(self) -> dict:
        return {
            "angles_deg": self.current_angles.copy(),
            "bus_voltage_v": 5.0,
            "connected": self.is_connected,
        }

    def emergency_stop(self):
        logger.warning("PCA9685 EMERGENCY STOP: Disabling all channels.")
        for ch in range(16):
            self.set_pwm(ch, 0, 0)

    def close(self):
        if self.i2c_bus:
            try:
                self.emergency_stop()
                self.i2c_bus.close()
            except Exception:
                pass
