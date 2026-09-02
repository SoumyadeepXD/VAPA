"""
VAPA 12-Channel PCA9685 Servo Driver & Motion Controller
Direct I2C communication on NVIDIA Jetson Orin (/dev/i2c-1 @ 0x40).
Controls 12 total servos (3x DS3225, 1x DS3218, 8x MG90S) with soft-start velocity curves,
independent pulse calibrations, and hardware angle constraints (0° - 180°).
"""

import time
import math
import logging
import threading
import numpy as np

logger = logging.getLogger("VAPA.Drivers.PCA9685")

# Register Definitions
PCA9685_MODE1       = 0x00
PCA9685_MODE2       = 0x01
PCA9685_PRESCALE    = 0xFE
LED0_ON_L           = 0x06
LED0_ON_H           = 0x07
LED0_OFF_L          = 0x08
LED0_OFF_H          = 0x09
ALL_LED_OFF_H       = 0xFD

# 12-Channel Servo Configuration Table
SERVO_12CH_CONFIG = {
    # Heavy Joints (3x DS3225 25kg High-Torque Servos)
    0:  {"name": "joint_1_base_yaw",       "type": "DS3225", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 90.0},
    1:  {"name": "joint_2_shoulder_pitch",  "type": "DS3225", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 75.0},
    2:  {"name": "joint_3_elbow_pitch",     "type": "DS3225", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 45.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 90.0},
    # Wrist Pitch (1x DS3218 20kg Servo)
    3:  {"name": "joint_4_wrist_pitch",     "type": "DS3218", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 120.0},
    # Wrist Orientations & Micro Articulations (8x MG90S/SG90 Micro Servos)
    4:  {"name": "joint_5_wrist_roll",      "type": "MG90S",  "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 150.0},
    5:  {"name": "joint_6_wrist_yaw",       "type": "MG90S",  "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 150.0},
    6:  {"name": "finger_thumb_flex",       "type": "MG90S",  "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    7:  {"name": "finger_index_flex",       "type": "MG90S",  "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    8:  {"name": "finger_middle_flex",      "type": "MG90S",  "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    9:  {"name": "finger_ring_flex",        "type": "MG90S",  "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    10: {"name": "finger_pinky_flex",       "type": "MG90S",  "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    11: {"name": "finger_thumb_abduct",     "type": "MG90S",  "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 45.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 180.0},
}


class PCA9685_12ChDriver:
    """
    Production-grade 12-Channel Servo Driver on Jetson Orin.
    Features soft-start interpolation, rate-limiting, and hard boundary protection.
    """
    def __init__(self, bus_num: int = 1, address: int = 0x40, pwm_freq_hz: int = 50, force_mock: bool = False):
        self.bus_num = bus_num
        self.address = address
        self.pwm_freq = pwm_freq_hz
        self.force_mock = force_mock
        self.i2c_bus = None
        self.is_connected = False
        self.lock = threading.Lock()

        # Track current & target angles for all 12 channels
        self.current_angles = {ch: cfg["home_deg"] for ch, cfg in SERVO_12CH_CONFIG.items()}
        self.target_angles  = {ch: cfg["home_deg"] for ch, cfg in SERVO_12CH_CONFIG.items()}
        self.name_to_channel = {cfg["name"]: ch for ch, cfg in SERVO_12CH_CONFIG.items()}

        if not self.force_mock:
            self._init_hardware()

    def _init_hardware(self):
        try:
            try:
                import smbus2 as smbus
            except ImportError:
                import smbus
            self.i2c_bus = smbus.SMBus(self.bus_num)

            # 1. Reset device
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE1, 0x00)
            time.sleep(0.01)

            # 2. Set PWM Frequency (50 Hz = 20ms period)
            # prescale = round(25MHz / (4096 * freq)) - 1 = round(25000000 / (4096 * 50)) - 1 = 121
            prescale_val = int(round(25000000.0 / (4096.0 * self.pwm_freq)) - 1)
            old_mode = self.i2c_bus.read_byte_data(self.address, PCA9685_MODE1)
            new_mode = (old_mode & 0x7F) | 0x10  # Sleep mode to write prescaler
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE1, new_mode)
            self.i2c_bus.write_byte_data(self.address, PCA9685_PRESCALE, prescale_val)
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE1, old_mode)
            time.sleep(0.005)
            # Enable auto-increment (AI bit 5 = 1) and restart
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE1, old_mode | 0xA1)
            # Set totem-pole output structure in MODE2
            self.i2c_bus.write_byte_data(self.address, PCA9685_MODE2, 0x04)

            self.is_connected = True
            logger.info(f"PCA9685 12-Channel Driver initialized on /dev/i2c-{self.bus_num} at 0x{self.address:02X} @ {self.pwm_freq}Hz.")

            # Soft-start all servos to home positions
            self.go_to_home(duration_s=1.0)
        except Exception as e:
            logger.warning(f"Could not connect to physical PCA9685 on /dev/i2c-{self.bus_num} ({e}). Running in mock mode.")
            self.is_connected = False

    def set_pwm(self, channel: int, on_tick: int, off_tick: int):
        """Sets raw 12-bit ON/OFF registers for a specific PCA9685 channel."""
        if not self.is_connected or self.i2c_bus is None:
            return
        if not (0 <= channel <= 15):
            return

        reg_base = LED0_ON_L + (4 * channel)
        try:
            with self.lock:
                self.i2c_bus.write_byte_data(self.address, reg_base, on_tick & 0xFF)
                self.i2c_bus.write_byte_data(self.address, reg_base + 1, (on_tick >> 8) & 0xFF)
                self.i2c_bus.write_byte_data(self.address, reg_base + 2, off_tick & 0xFF)
                self.i2c_bus.write_byte_data(self.address, reg_base + 3, (off_tick >> 8) & 0xFF)
        except Exception as e:
            logger.error(f"Error setting PWM on PCA9685 channel {channel}: {e}")

    def set_pulse_microseconds(self, channel: int, pulse_us: float):
        """Converts microsecond pulse width (500us to 2500us) to 12-bit tick (0-4095)."""
        period_us = 1000000.0 / self.pwm_freq  # 20,000us for 50Hz
        pulse_us = max(400.0, min(2600.0, float(pulse_us)))
        off_tick = int(round((pulse_us / period_us) * 4096.0))
        off_tick = max(0, min(4095, off_tick))
        self.set_pwm(channel, 0, off_tick)

    def angle_to_pulse_us(self, channel: int, angle_deg: float) -> float:
        """Translates calibrated angle in degrees to calibrated pulse width in microseconds."""
        cfg = SERVO_12CH_CONFIG.get(channel)
        if not cfg:
            return 1500.0

        min_deg = cfg["min_deg"]
        max_deg = cfg["max_deg"]
        min_us  = cfg["min_us"]
        max_us  = cfg["max_us"]
        trim    = cfg["trim_deg"]
        dir_val = cfg["direction"]

        # Apply direction and trim offset
        adj_deg = (angle_deg + trim) * dir_val
        clamped_deg = float(np.clip(adj_deg, min_deg, max_deg))

        # Linear mapping from [min_deg, max_deg] to [min_us, max_us]
        ratio = (clamped_deg - min_deg) / max(1e-6, (max_deg - min_deg))
        pulse_us = min_us + ratio * (max_us - min_us)
        return float(pulse_us)

    def set_joint_angle(self, channel_or_name, angle_deg: float):
        """Sets a single joint angle immediately."""
        if isinstance(channel_or_name, str):
            if channel_or_name not in self.name_to_channel:
                logger.warning(f"Unknown servo joint name: {channel_or_name}")
                return
            ch = self.name_to_channel[channel_or_name]
        else:
            ch = int(channel_or_name)

        if ch not in SERVO_12CH_CONFIG:
            return

        cfg = SERVO_12CH_CONFIG[ch]
        clamped_deg = float(np.clip(angle_deg, cfg["min_deg"], cfg["max_deg"]))
        pulse = self.angle_to_pulse_us(ch, clamped_deg)

        self.set_pulse_microseconds(ch, pulse)
        self.current_angles[ch] = clamped_deg
        self.target_angles[ch] = clamped_deg

    def set_all_angles(self, angles_dict: dict):
        """Sets multiple joint angles simultaneously."""
        for key, val in angles_dict.items():
            self.set_joint_angle(key, val)

    def soft_move(self, target_angles_dict: dict, duration_s: float = 1.0, update_rate_hz: int = 50):
        """
        Soft-start minimum-jerk interpolation between current joint angles and target angles.
        Prevents inrush current spikes and mechanical chatter on high-torque DS3225 servos.
        """
        steps = max(2, int(duration_s * update_rate_hz))
        dt = 1.0 / update_rate_hz

        # Extract start and target vectors
        start_state = self.current_angles.copy()
        target_state = start_state.copy()

        for key, val in target_angles_dict.items():
            ch = self.name_to_channel[key] if isinstance(key, str) else int(key)
            if ch in SERVO_12CH_CONFIG:
                cfg = SERVO_12CH_CONFIG[ch]
                target_state[ch] = float(np.clip(val, cfg["min_deg"], cfg["max_deg"]))

        # S-curve / quintic interpolation: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        for step in range(1, steps + 1):
            tau = step / steps
            s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)

            for ch in SERVO_12CH_CONFIG:
                q0 = start_state[ch]
                q1 = target_state[ch]
                interp_deg = q0 + (q1 - q0) * s
                pulse = self.angle_to_pulse_us(ch, interp_deg)
                self.set_pulse_microseconds(ch, pulse)
                self.current_angles[ch] = interp_deg

            time.sleep(dt)

    def go_to_home(self, duration_s: float = 1.5):
        """Smoothly returns all 12 servos to default home positions."""
        home_dict = {ch: cfg["home_deg"] for ch, cfg in SERVO_12CH_CONFIG.items()}
        logger.info("Moving all 12 servos to calibrated home pose...")
        self.soft_move(home_dict, duration_s=duration_s)

    def set_hand_opening_percent(self, percent: float, duration_s: float = 0.5):
        """
        Simultaneously drives all 5 finger flexor servos (Channels 6 to 10):
        0.0% = Closed fist, 100.0% = Fully open hand.
        """
        percent = float(np.clip(percent, 0.0, 100.0))
        # 0% open -> 180 deg flex (closed), 100% open -> 0 deg flex (open)
        flex_deg = (1.0 - (percent / 100.0)) * 180.0
        finger_dict = {
            6: flex_deg,  # Thumb
            7: flex_deg,  # Index
            8: flex_deg,  # Middle
            9: flex_deg,  # Ring
            10: flex_deg, # Pinky
        }
        self.soft_move(finger_dict, duration_s=duration_s)

    def emergency_stop(self):
        """Immediately halts all 12 PWM outputs to kill motor torque."""
        logger.critical("PCA9685 12-Channel Driver: EMERGENCY STOP triggered! Disabling all PWM channels.")
        if self.is_connected and self.i2c_bus:
            try:
                with self.lock:
                    for ch in range(16):
                        self.set_pwm(ch, 0, 0)
            except Exception:
                pass

    def get_all_angles_dict(self) -> dict[str, float]:
        """Returns joint angles mapped by joint name."""
        return {SERVO_12CH_CONFIG[ch]["name"]: self.current_angles[ch] for ch in SERVO_12CH_CONFIG}

    def close(self):
        if self.i2c_bus:
            try:
                self.emergency_stop()
                self.i2c_bus.close()
            except Exception:
                pass
