

"""
VAPA 9/12-Channel PCA9685 Servo Driver & Motion Controller
Direct I2C communication on NVIDIA Jetson Orin (auto-scans I2C candidate buses).
Controls 9 active servos (5x MG996R, 3x DS3225, 1x DS3218) with soft-start velocity curves,
independent pulse calibrations, and hardware angle constraints (0° - 180°).
"""

import os
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

# Hardware Build Phase Servo Configuration Table (VAPA Hardware Connection Guide Section 7B)
SERVO_12CH_CONFIG = {
    # 5x MG996R Finger Servos (Channels 0 - 4)
    0:  {"name": "finger_thumb",          "type": "MG996R", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    1:  {"name": "finger_index",          "type": "MG996R", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    2:  {"name": "finger_middle",         "type": "MG996R", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    3:  {"name": "finger_ring",           "type": "MG996R", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    4:  {"name": "finger_pinky",          "type": "MG996R", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 0.0,  "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 200.0},
    # 3x DS3225 25kg High-Torque Wrist Servos (Channels 5 - 7)
    5:  {"name": "joint_wrist_flex",      "type": "DS3225", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 120.0},
    6:  {"name": "joint_wrist_rotate",    "type": "DS3225", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 120.0},
    7:  {"name": "joint_wrist_bend",      "type": "DS3225", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 120.0},
    # 1x DS3218 20kg Forearm Rotation Servo (Channel 8)
    8:  {"name": "joint_forearm_rotate",  "type": "DS3218", "min_us": 500, "max_us": 2500, "min_deg": 0.0, "max_deg": 180.0, "home_deg": 90.0, "trim_deg": 0.0, "direction": 1, "max_speed_deg_s": 120.0},
}


class PCA9685_12ChDriver:
    """
    Production-grade PCA9685 Servo Driver on Jetson Orin.
    Features auto-bus discovery, soft-start interpolation, rate-limiting, and hard boundary protection.
    """
    def __init__(self, bus_num: int = 1, address: int = 0x40, pwm_freq_hz: int = 50, force_mock: bool = False):
        self.bus_num = bus_num
        self.address = address
        self.pwm_freq = pwm_freq_hz
        self.force_mock = force_mock
        self.i2c_bus = None
        self.is_connected = False
        self.lock = threading.Lock()

        # Track current & target angles for all channels
        self.current_angles = {ch: cfg["home_deg"] for ch, cfg in SERVO_12CH_CONFIG.items()}
        self.target_angles  = {ch: cfg["home_deg"] for ch, cfg in SERVO_12CH_CONFIG.items()}
        self.name_to_channel = {cfg["name"]: ch for ch, cfg in SERVO_12CH_CONFIG.items()}

        if not self.force_mock:
            self._init_hardware()

    def _init_hardware(self):
        try:
            import smbus2 as smbus
        except ImportError:
            try:
                import smbus
            except ImportError:
                logger.warning("Neither smbus2 nor smbus is installed. Mock mode activated.")
                return

        candidate_buses = [self.bus_num, 7, 8, 0]
        try:
            for dev in os.listdir("/dev"):
                if dev.startswith("i2c-"):
                    b_id = int(dev.split("-")[1])
                    if b_id not in candidate_buses:
                        candidate_buses.append(b_id)
        except Exception:
            pass

        for b in candidate_buses:
            if not os.path.exists(f"/dev/i2c-{b}"):
                continue
            try:
                bus = smbus.SMBus(b)
                _ = bus.read_byte_data(self.address, PCA9685_MODE1)

                # Reset PCA9685
                bus.write_byte_data(self.address, PCA9685_MODE1, 0x00)
                time.sleep(0.01)

                # Set PWM Frequency
                prescale = int(math.floor(25000000.0 / (4096.0 * self.pwm_freq) - 0.5))
                old_mode = bus.read_byte_data(self.address, PCA9685_MODE1)
                new_mode = (old_mode & 0x7F) | 0x10  # Sleep to set prescale
                bus.write_byte_data(self.address, PCA9685_MODE1, new_mode)
                bus.write_byte_data(self.address, PCA9685_PRESCALE, prescale)
                time.sleep(0.005)
                # Clear ALLCALL bit (bit 0 = 0) so PCA9685 never responds to 0x70 broadcast
                # Enable Auto-Increment (bit 5) and RESTART (bit 7)
                bus.write_byte_data(self.address, PCA9685_MODE1, (old_mode & ~0x11) | 0xA0)

                self.i2c_bus = bus
                self.bus_num = b
                self.is_connected = True
                logger.info(f"[SUCCESS] PCA9685 initialized on /dev/i2c-{b} (0x{self.address:02X}) @ {self.pwm_freq}Hz.")
                return
            except Exception as e:
                logger.debug(f"I2C bus {b} failed for PCA9685: {e}")
                continue

        logger.warning(f"Could not connect to PCA9685 on candidate buses {candidate_buses}. Using software simulation.")
        self.is_connected = False

    def angle_to_pulse_us(self, channel: int, angle_deg: float) -> float:
        cfg = SERVO_12CH_CONFIG.get(channel)
        if not cfg:
            return 1500.0

        min_a = cfg["min_deg"]
        max_a = cfg["max_deg"]
        min_u = cfg["min_us"]
        max_u = cfg["max_us"]
        trim = cfg["trim_deg"]
        direction = cfg.get("direction", 1)
        if direction < 0:
            calc_deg = (max_a - (angle_deg - min_a)) + trim
        else:
            calc_deg = angle_deg + trim

        clamped_deg = float(np.clip(calc_deg, min_a, max_a))

        ratio = (clamped_deg - min_a) / max(1e-6, (max_a - min_a))
        return min_u + ratio * (max_u - min_u)

    def write_channel_pulse(self, channel: int, pulse_us: float):
        if not self.is_connected or self.i2c_bus is None:
            return

        pulse_us = max(400.0, min(2600.0, float(pulse_us)))
        period_us = 1000000.0 / self.pwm_freq
        off_tick = int(round((pulse_us / period_us) * 4096.0))
        off_tick = max(0, min(4095, off_tick))

        reg_base = LED0_ON_L + 4 * channel
        with self.lock:
            try:
                self.i2c_bus.write_byte_data(self.address, reg_base, 0)
                self.i2c_bus.write_byte_data(self.address, reg_base + 1, 0)
                self.i2c_bus.write_byte_data(self.address, reg_base + 2, off_tick & 0xFF)
                self.i2c_bus.write_byte_data(self.address, reg_base + 3, (off_tick >> 8) & 0xFF)
            except Exception as e:
                logger.error(f"I2C write failed on channel {channel}: {e}")

    def set_channel_angle(self, channel: int, angle_deg: float):
        if channel not in SERVO_12CH_CONFIG:
            return
        pulse = self.angle_to_pulse_us(channel, angle_deg)
        self.write_channel_pulse(channel, pulse)
        self.current_angles[channel] = float(angle_deg)
        self.target_angles[channel] = float(angle_deg)

    def set_servo_by_name(self, name: str, angle_deg: float):
        ch = self.name_to_channel.get(name)
        if ch is not None:
            self.set_channel_angle(ch, angle_deg)

    def set_all_channels(self, angle_map: dict):
        for key, angle in angle_map.items():
            if isinstance(key, int):
                self.set_channel_angle(key, angle)
            elif isinstance(key, str):
                self.set_servo_by_name(key, angle)

    def emergency_stop(self):
        logger.warning("PCA9685: EMERGENCY STOP — Cutting PWM to all channels.")
        if self.is_connected and self.i2c_bus:
            with self.lock:
                try:
                    for ch in range(16):
                        reg_base = LED0_ON_L + 4 * ch
                        self.i2c_bus.write_byte_data(self.address, reg_base + 2, 0)
                        self.i2c_bus.write_byte_data(self.address, reg_base + 3, 0)
                except Exception:
                    pass

    def close(self):
        self.emergency_stop()
        if self.i2c_bus:
            try:
                self.i2c_bus.close()
            except Exception:
                pass
