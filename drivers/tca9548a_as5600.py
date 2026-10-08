"""
VAPA TCA9548A I2C Multiplexer & AS5600 12-bit Magnetic Rotary Encoder Driver
Enables polling multiple AS5600 magnetic encoders sharing identical I2C address (0x36)
via the TCA9548A Multiplexer (0x70) on NVIDIA Jetson Orin (auto-scans candidate I2C buses).
Matches Hardware Connection Guide Section 9 (Channels 0-3).
"""

import os
import time
import math
import logging
import numpy as np

logger = logging.getLogger("VAPA.Drivers.AS5600")

# I2C Addresses
TCA9548A_DEFAULT_ADDR = 0x70  # A0=GND, A1=GND, A2=GND
AS5600_I2C_ADDR       = 0x36  # Fixed factory address for AS5600

# AS5600 Register Map
AS5600_REG_STATUS      = 0x0B  # Magnet status (MD, ML, MH)
AS5600_REG_RAW_ANGLE_H = 0x0C  # Raw Angle 11:8
AS5600_REG_RAW_ANGLE_L = 0x0D  # Raw Angle 7:0
AS5600_REG_ANGLE_H     = 0x0E  # Scaled Angle 11:8
AS5600_REG_ANGLE_L     = 0x0F  # Scaled Angle 7:0
AS5600_REG_CONF_H      = 0x07  # Configuration register
AS5600_REG_CONF_L      = 0x08

# Channel Mapping for Prosthetic Joints (Hardware Connection Guide Section 9A)
ENCODER_CHANNEL_MAP = {
    0: {"name": "finger_group_angle",  "description": "Finger Group Flexion Encoder", "zero_offset_deg": 0.0, "direction": 1},
    1: {"name": "wrist_flex_angle",    "description": "Wrist Flexion/Pitch Encoder",  "zero_offset_deg": 0.0, "direction": 1},
    2: {"name": "wrist_rotate_angle",  "description": "Wrist Pronation/Supination",   "zero_offset_deg": 0.0, "direction": 1},
    3: {"name": "forearm_rotate_angle", "description": "Forearm Rotation Encoder",    "zero_offset_deg": 0.0, "direction": 1},
}


class AS5600EncoderMux:
    """
    Manages 4x AS5600 12-bit Magnetic Encoders via TCA9548A I2C Multiplexer.
    """
    def __init__(self, bus_num: int = 1, mux_address: int = TCA9548A_DEFAULT_ADDR, force_mock: bool = False):
        self.bus_num = bus_num
        self.mux_address = mux_address
        self.force_mock = force_mock
        self.i2c_bus = None
        self.is_connected = False
        self.last_angles_deg = {ch: 0.0 for ch in ENCODER_CHANNEL_MAP}
        self.last_raw_counts = {ch: 0 for ch in ENCODER_CHANNEL_MAP}

        if not self.force_mock:
            self._init_i2c()

    def _init_i2c(self):
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
                # Test multiplexer presence by writing to MUX
                bus.write_byte(self.mux_address, 0x01)
                self.i2c_bus = bus
                self.bus_num = b
                self.is_connected = True
                logger.info(f"[SUCCESS] TCA9548A Multiplexer (0x{self.mux_address:02X}) initialized on /dev/i2c-{b}.")
                return
            except Exception as e:
                logger.debug(f"I2C bus {b} failed for TCA9548A: {e}")
                continue

        logger.warning(f"Could not connect to TCA9548A on candidate buses {candidate_buses}. Using software simulation.")
        self.is_connected = False

    def select_mux_channel(self, channel: int):
        """
        Enables a single channel (0-7) on the TCA9548A multiplexer.
        Writing 1 << channel enables that channel; writing 0 disables all channels.
        """
        if not self.is_connected or self.i2c_bus is None:
            return
        if not (0 <= channel <= 7):
            raise ValueError(f"Invalid TCA9548A channel {channel}. Must be 0-7.")

        control_byte = 1 << channel
        try:
            self.i2c_bus.write_byte(self.mux_address, control_byte)
        except Exception as e:
            logger.error(f"Error selecting TCA9548A MUX channel {channel}: {e}")

    def read_raw_angle_12bit(self, channel: int) -> int:
        """
        Switches MUX to given channel and reads 12-bit raw angle (0 - 4095) from AS5600.
        """
        if not self.is_connected or self.i2c_bus is None:
            # Simulated angle reading
            t = time.time()
            sim_counts = int((math.sin(t * 0.5 + channel) * 0.5 + 0.5) * 4095)
            return sim_counts

        try:
            # 1. Switch MUX to target channel
            self.select_mux_channel(channel)
            time.sleep(0.0005)  # 500us settle time

            # 2. Read 2 bytes from RAW_ANGLE registers (0x0C and 0x0D)
            msb = self.i2c_bus.read_byte_data(AS5600_I2C_ADDR, AS5600_REG_RAW_ANGLE_H)
            lsb = self.i2c_bus.read_byte_data(AS5600_I2C_ADDR, AS5600_REG_RAW_ANGLE_L)

            raw_12bit = ((msb & 0x0F) << 8) | (lsb & 0xFF)
            return raw_12bit
        except Exception as e:
            logger.error(f"Error reading AS5600 on MUX channel {channel}: {e}")
            return self.last_raw_counts.get(channel, 0)

    def read_angle_degrees(self, channel: int) -> float:
        """
        Reads 12-bit raw angle, applies zero-offset trim & direction,
        and returns calibrated angle in degrees (0.0° - 360.0°).
        """
        raw_counts = self.read_raw_angle_12bit(channel)
        self.last_raw_counts[channel] = raw_counts

        raw_deg = (float(raw_counts) / 4096.0) * 360.0

        cfg = ENCODER_CHANNEL_MAP.get(channel, {"zero_offset_deg": 0.0, "direction": 1})
        zero_offset = cfg.get("zero_offset_deg", 0.0)
        direction = cfg.get("direction", 1)

        cal_deg = (raw_deg * direction) - zero_offset
        cal_deg = cal_deg % 360.0
        if cal_deg < 0.0:
            cal_deg += 360.0

        self.last_angles_deg[channel] = cal_deg
        return cal_deg

    def read_all_encoders(self) -> dict[str, float]:
        """
        Polls all 4 configured AS5600 encoders across MUX channels 0 to 3.
        Returns dictionary mapping joint names to calibrated angles in degrees.
        """
        results = {}
        for ch, cfg in ENCODER_CHANNEL_MAP.items():
            name = cfg["name"]
            deg = self.read_angle_degrees(ch)
            results[name] = deg
        return results

    def check_magnet_status(self, channel: int) -> dict:
        """
        Reads AS5600 status register (0x0B) to verify magnetic field strength:
        - MD (Bit 5): Magnet Detected (1 = OK)
        - ML (Bit 4): Magnet Too Weak (1 = Alert)
        - MH (Bit 3): Magnet Too Strong (1 = Alert)
        """
        if not self.is_connected or self.i2c_bus is None:
            return {"magnet_detected": True, "too_weak": False, "too_strong": False}

        try:
            self.select_mux_channel(channel)
            status_byte = self.i2c_bus.read_byte_data(AS5600_I2C_ADDR, AS5600_REG_STATUS)
            return {
                "magnet_detected": bool(status_byte & 0x20),
                "too_weak": bool(status_byte & 0x10),
                "too_strong": bool(status_byte & 0x08),
            }
        except Exception as e:
            logger.error(f"Error checking magnet status on channel {channel}: {e}")
            return {"magnet_detected": False, "too_weak": False, "too_strong": False}

    def close(self):
        if self.i2c_bus:
            try:
                self.i2c_bus.write_byte(self.mux_address, 0x00)
                self.i2c_bus.close()
            except Exception:
                pass
