"""
VAPA Serial Bus Servo Driver & Microcontroller Bridge
Communicates with smart bus servos (Feetech STS/SCS, Dynamixel, Hiwonder)
and Arduino/Teensy servo bridges over UART / USB-Serial.
"""

import time
import logging
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from actuation.servo_interface import BaseServoDriver
from config.hardware_config import (
    SERIAL_SERVO_PORT,
    SERIAL_SERVO_BAUD_RATE,
    SERVO_CHANNELS,
)

logger = logging.getLogger("VAPA.Actuation.SerialServo")


class SerialServoDriver(BaseServoDriver):
    """Controls multi-servo arm joints via Serial UART / USB bus."""
    def __init__(self, port=SERIAL_SERVO_PORT, baud_rate=SERIAL_SERVO_BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.current_angles = {}
        self.is_connected = False

        self._init_serial()

    def _init_serial(self):
        if not SERIAL_AVAILABLE:
            logger.warning("pyserial not available. SerialServoDriver running in disconnected mode.")
            return
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=0.05)
            self.is_connected = True
            logger.info(f"Connected to Serial Bus Servo Controller on {self.port} @ {self.baud_rate} baud.")
        except Exception as e:
            logger.warning(f"Could not open serial servo port {self.port}: {e}")
            self.is_connected = False

    def set_servo_pulse(self, channel: int, pulse_us: float):
        if self.serial_conn and self.serial_conn.is_open:
            cmd = f"P,{channel},{int(pulse_us)}\n"
            try:
                self.serial_conn.write(cmd.encode("utf-8"))
            except Exception as e:
                logger.error(f"Serial write error: {e}")

    def set_joint_angle(self, joint_name: str, angle_deg: float):
        cfg = SERVO_CHANNELS.get(joint_name)
        if not cfg:
            return

        ch = cfg["channel"]
        self.current_angles[joint_name] = float(angle_deg)
        if self.serial_conn and self.serial_conn.is_open:
            cmd = f"A,{ch},{angle_deg:.2f}\n"
            try:
                self.serial_conn.write(cmd.encode("utf-8"))
            except Exception as e:
                logger.error(f"Serial servo angle send error: {e}")

    def set_all_angles(self, angles_deg: dict[str, float]):
        for j_name, angle in angles_deg.items():
            self.set_joint_angle(j_name, angle)

    def read_feedback(self) -> dict:
        return {
            "angles_deg": self.current_angles.copy(),
            "connected": self.is_connected,
        }

    def emergency_stop(self):
        logger.warning("SERIAL SERVO EMERGENCY STOP: Sending torque disable.")
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(b"ESTOP\n")
            except Exception:
                pass

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.emergency_stop()
                self.serial_conn.close()
            except Exception:
                pass
