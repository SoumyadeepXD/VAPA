"""
VAPA PCA9685 I2C 16-Channel 12-bit PWM Servo Driver
Direct I2C communication on NVIDIA Jetson Orin (auto-probes bus 1, 7, 8, 0).
Supports calibrated angle-to-PWM translation for:
- 5x MG996R Finger Servos (Channels 0-4)
- 3x DS3225 Wrist Servos (Channels 5-7: Flex, Rotate, Bend)
- 1x DS3218 Forearm Rotation Servo (Channel 8)
"""

import os
import time
import math
import logging
import numpy as np

from actuation.servo_interface import BaseServoDriver
from config.hardware_config import (
    PCA9685_I2C_BUS,
    PCA9685_I2C_ADDRESS,
    PCA9685_PWM_FREQ_HZ,
    JETSON_I2C_CANDIDATE_BUSES,
    SERVO_CHANNELS,
    SERVO_ALIASES,
)

logger = logging.getLogger("VAPA.Actuation.PCA9685")

PCA9685_MODE1 = 0x00
PCA9685_MODE2 = 0x01
PCA9685_PRESCALE = 0xFE
LED0_ON_L = 0x06
LED0_ON_H = 0x07
LED0_OFF_L = 0x08
LED0_OFF_H = 0x09
ALL_LED_OFF_H = 0xFD


class PCA9685ServoDriver(BaseServoDriver):
    """Controls 9-servo prosthetic arm & hand via PCA9685 I2C on Jetson Orin."""
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
            import smbus2 as smbus
        except ImportError:
            try:
                import smbus
            except ImportError:
                logger.warning("Neither smbus2 nor smbus is installed. Hardware I2C unavailable.")
                self.is_connected = False
                return

        # Build list of candidate buses starting with specified bus_num, then known Jetson buses
        candidate_buses = [self.bus_num] + [b for b in JETSON_I2C_CANDIDATE_BUSES if b != self.bus_num]

        # Also search any /dev/i2c-* devices present in /dev
        try:
            for dev in os.listdir("/dev"):
                if dev.startswith("i2c-"):
                    try:
                        num = int(dev.split("-")[1])
                        if num not in candidate_buses:
                            candidate_buses.append(num)
                    except ValueError:
                        pass
        except Exception:
            pass

        for bus_id in candidate_buses:
            dev_path = f"/dev/i2c-{bus_id}"
            if not os.path.exists(dev_path):
                continue

            try:
                bus = smbus.SMBus(bus_id)
                # Try reading MODE1 register
                _ = bus.read_byte_data(self.address, PCA9685_MODE1)

                # Reset PCA9685 MODE1
                bus.write_byte_data(self.address, PCA9685_MODE1, 0x00)
                time.sleep(0.01)

                # Set PWM Frequency (50Hz)
                prescale_val = int(math.floor(25000000.0 / (4096.0 * self.freq_hz) - 0.5))
                old_mode = bus.read_byte_data(self.address, PCA9685_MODE1)
                new_mode = (old_mode & 0x7F) | 0x10  # Sleep mode to configure prescaler
                bus.write_byte_data(self.address, PCA9685_MODE1, new_mode)
                bus.write_byte_data(self.address, PCA9685_PRESCALE, prescale_val)
                time.sleep(0.005)
                # Clear ALLCALL bit (bit 0 = 0) to prevent broadcast collision at 0x70
                # Enable Auto-Increment (bit 5) and RESTART (bit 7)
                bus.write_byte_data(self.address, PCA9685_MODE1, (old_mode & ~0x11) | 0xA0)

                self.i2c_bus = bus
                self.bus_num = bus_id
                self.is_connected = True
                logger.info(f"[SUCCESS] PCA9685 connected on /dev/i2c-{bus_id} at address 0x{self.address:02X} @ {self.freq_hz}Hz.")
                return

            except Exception as e:
                logger.debug(f"Probing bus {bus_id} at 0x{self.address:02X} failed: {e}")
                continue

        logger.warning(
            f"PCA9685 not detected at 0x{self.address:02X} across candidate I2C buses {candidate_buses}. "
            f"Check wiring: Jetson Pin 1 (3.3V) -> PCA9685 VCC, Pin 3 -> SDA, Pin 5 -> SCL, Pin 6 -> GND, "
            f"6V Rail -> V+, and check group permissions ('sudo usermod -a -G i2c $USER')."
        )
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
        # Resolve aliases
        canonical_name = SERVO_ALIASES.get(joint_name, joint_name)
        cfg = SERVO_CHANNELS.get(canonical_name)
        if not cfg:
            return 1500.0

        min_a = cfg["min_angle_deg"]
        max_a = cfg["max_angle_deg"]
        min_u = cfg["min_us"]
        max_u = cfg["max_us"]
        offset = cfg.get("center_offset_deg", 0.0)
        direction = cfg.get("direction", 1)

        if direction < 0:
            adj_angle = (max_a - (angle_deg - min_a)) + offset
        else:
            adj_angle = angle_deg + offset

        clamped_angle = np.clip(adj_angle, min_a, max_a)

        ratio = (clamped_angle - min_a) / max(1e-6, (max_a - min_a))
        pulse_us = min_u + ratio * (max_u - min_u)
        return float(pulse_us)

    def set_joint_angle(self, joint_name: str, angle_deg: float):
        # Special handler for master gripper: maps to 5 fingers (CH 0-4)
        if joint_name == "joint_6_gripper":
            self.set_hand_opening_percent(angle_deg)
            return

        canonical_name = SERVO_ALIASES.get(joint_name, joint_name)
        if canonical_name not in SERVO_CHANNELS:
            logger.debug(f"Joint '{joint_name}' (canonical: '{canonical_name}') not in active 9-servo map.")
            return

        cfg = SERVO_CHANNELS[canonical_name]
        ch = cfg["channel"]
        pulse = self.angle_to_pulse_us(canonical_name, angle_deg)
        self.set_servo_pulse(ch, pulse)
        self.current_angles[joint_name] = float(angle_deg)
        self.current_angles[canonical_name] = float(angle_deg)

    def set_hand_opening_percent(self, percent: float):
        """
        Drives all 5 fingers simultaneously (CH 0 to CH 4).
        0.0% = Closed (180 deg flexion), 100.0% = Open (0 deg flexion).
        """
        percent = float(np.clip(percent, 0.0, 100.0))
        # 100% open = 0 deg, 0% open = 180 deg
        flexion_deg = (1.0 - (percent / 100.0)) * 180.0

        for f_name in ("finger_thumb", "finger_index", "finger_middle", "finger_ring", "finger_pinky"):
            self.set_joint_angle(f_name, flexion_deg)
        self.current_angles["joint_6_gripper"] = percent

    def set_all_angles(self, angles_deg: dict[str, float]):
        for j_name, angle in angles_deg.items():
            self.set_joint_angle(j_name, angle)

    def read_feedback(self) -> dict:
        return {
            "angles_deg": self.current_angles.copy(),
            "bus_voltage_v": 6.0,
            "connected": self.is_connected,
            "bus_num": self.bus_num,
        }

    def emergency_stop(self):
        logger.warning("PCA9685 EMERGENCY STOP: Shutting down all PWM channels.")
        for ch in range(16):
            self.set_pwm(ch, 0, 0)

    def close(self):
        if self.i2c_bus:
            try:
                self.emergency_stop()
                self.i2c_bus.close()
            except Exception:
                pass
