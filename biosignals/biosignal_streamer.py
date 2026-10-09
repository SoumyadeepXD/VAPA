"""
VAPA Biosignal Streamer Interface
Connects to ESP32 Node 2 via AsyncESP32Receiver (over Jetson UART /dev/ttyTHS1, /dev/ttyTHS0, /dev/ttyUSB0)
or provides a realistic synthetic bio-signal generator for offline testing.
Provides both multi-channel EMG/EEG arrays for DSP decoders and real-time 5-finger FSR contact forces.
"""

import os
import time
import math
import logging
import threading
import numpy as np

from config.hardware_config import (
    BIOSIGNAL_SERIAL_PORT,
    BIOSIGNAL_BAUD_RATE,
    NUM_EMG_CHANNELS,
    NUM_EEG_CHANNELS,
    NUM_FSR_CHANNELS,
)
from config.system_config import EMG_SAMPLING_RATE_HZ, EEG_SAMPLING_RATE_HZ
from biosignals.esp32_serial_receiver import AsyncESP32Receiver, ESP32TelemetryFrame

logger = logging.getLogger("VAPA.Biosignals.Streamer")


class SyntheticBiosignalStreamer:
    """
    Generates realistic physiological multi-channel EMG and EEG waveforms.
    Allows injecting synthetic muscle contractions and motor imagery for testing.
    """
    def __init__(self, num_emg=4, num_eeg=4):
        self.num_emg = num_emg
        self.num_eeg = num_eeg
        self.start_time = time.time()

        # State overrides for interactive testing
        self.inject_emg_grasp = False
        self.inject_emg_open = False
        self.inject_emg_estop = False
        self.inject_eeg_reach = False
        self.inject_eeg_cycle = False

        self.phase = 0.0
        self.emg_phase = 0.0
        self.eeg_phase = 0.0
        logger.info("SyntheticBiosignalStreamer initialized.")

    def trigger_gesture(self, gesture_name: str, duration_s: float = 1.0):
        def _reset_after(prop_name, d):
            time.sleep(d)
            setattr(self, prop_name, False)

        gesture_map = {
            "grasp": "inject_emg_grasp",
            "flexor": "inject_emg_grasp",
            "flex": "inject_emg_grasp",
            "fist": "inject_emg_grasp",
            "open": "inject_emg_open",
            "extensor": "inject_emg_open",
            "ext": "inject_emg_open",
            "estop": "inject_emg_estop",
            "co_contraction": "inject_emg_estop",
            "reach": "inject_eeg_reach",
            "cycle": "inject_eeg_cycle",
        }
        if gesture_name == "rest":
            self.inject_emg_grasp = False
            self.inject_emg_open = False
            self.inject_emg_estop = False
            return
        if gesture_name in gesture_map:
            attr = gesture_map[gesture_name]
            setattr(self, attr, True)
            if gesture_name in ("estop", "co_contraction"):
                self.inject_emg_grasp = False
                self.inject_emg_open = False
            t = threading.Thread(target=_reset_after, args=(attr, duration_s), daemon=True)
            t.start()
            logger.info(f"Triggered synthetic gesture: {gesture_name} for {duration_s}s")

    def read_chunk(self, num_samples: int = 20):
        t_emg = np.linspace(self.emg_phase, self.emg_phase + num_samples / EMG_SAMPLING_RATE_HZ, num_samples, endpoint=False)
        self.emg_phase += num_samples / EMG_SAMPLING_RATE_HZ
        self.phase = self.emg_phase

        # Base EMG (Raw microvolts: White noise + 50Hz hum + baseline muscle tone)
        emg_data = np.random.normal(0, 3.0, (self.num_emg, num_samples))
        emg_data += 2.0 * np.sin(2 * np.pi * 50.0 * t_emg)

        if self.inject_emg_estop:
            burst = np.random.normal(0, 150.0, num_samples)
            emg_data[0] += burst
            emg_data[1] += burst
        elif self.inject_emg_grasp:
            burst = np.random.normal(0, 120.0, num_samples) * (1.0 + 0.5 * np.sin(2 * np.pi * 150 * t_emg))
            emg_data[0] += burst
        elif self.inject_emg_open:
            burst = np.random.normal(0, 110.0, num_samples) * (1.0 + 0.5 * np.sin(2 * np.pi * 180 * t_emg))
            emg_data[1] += burst

        t_eeg = np.linspace(self.eeg_phase, self.eeg_phase + num_samples / EEG_SAMPLING_RATE_HZ, num_samples, endpoint=False)
        self.eeg_phase += num_samples / EEG_SAMPLING_RATE_HZ

        # Base EEG
        eeg_data = np.random.normal(0, 4.0, (self.num_eeg, num_samples))
        if not self.inject_eeg_reach:
            eeg_data[0] += 25.0 * np.sin(2 * np.pi * 10.0 * t_eeg)
            eeg_data[2] += 20.0 * np.sin(2 * np.pi * 10.0 * t_eeg)
        else:
            eeg_data[0] += 4.0 * np.sin(2 * np.pi * 10.0 * t_eeg)
            eeg_data[0] += 18.0 * np.sin(2 * np.pi * 22.0 * t_eeg)

        if self.inject_eeg_cycle:
            eeg_data[3] += 80.0

        return emg_data, eeg_data


class BiosignalStreamer:
    """
    Unified hardware biosignal and tactile streamer.
    Bridges ESP32 UART telemetry (5x FSR, 1x MyoWare EMG, 1x EEG) to VAPA decoders.
    """
    def __init__(self, port=BIOSIGNAL_SERIAL_PORT, baud_rate=BIOSIGNAL_BAUD_RATE, force_mock=False):
        self.port = port
        self.baud_rate = baud_rate
        self.force_mock = force_mock

        self.synthetic_streamer = SyntheticBiosignalStreamer()
        self.esp32_receiver = AsyncESP32Receiver(port=self.port, baud_rate=self.baud_rate, force_mock=force_mock)

        if not self.force_mock:
            self.esp32_receiver.start()

    @property
    def is_connected(self) -> bool:
        return self.esp32_receiver.is_connected

    @property
    def is_synthetic(self) -> bool:
        return not self.is_connected

    def read_chunk(self, num_samples: int = 10):
        """
        Returns (emg_data, eeg_data) arrays of shape (4, num_samples).
        If ESP32 hardware is connected, populates channel 0 with live MyoWare EMG / EEG telemetry.
        """
        if self.esp32_receiver.is_connected:
            frame = self.esp32_receiver.get_latest_frame()

            # Construct 4-channel arrays for EMGDecoder and EEGDecoder compatibility
            emg_data = np.random.normal(0, 1.0, (4, num_samples))
            eeg_data = np.random.normal(0, 1.0, (4, num_samples))

            # Channel 0: Scaled MyoWare EMG flexor signal (volts * 80 to map to microvolt-scale DSP)
            emg_data[0, :] = float(frame.emg_flex_volts * 80.0) + np.random.normal(0, 2.0, num_samples)

            # Channel 1: Scaled MyoWare EMG extensor signal
            emg_data[1, :] = float(frame.emg_ext_volts * 80.0) + np.random.normal(0, 2.0, num_samples)

            # Channel 0: EEG signal
            eeg_data[0, :] = float(frame.eeg_volts * 50.0) + np.random.normal(0, 3.0, num_samples)

            return emg_data, eeg_data

        # Fallback to synthetic streamer
        return self.synthetic_streamer.read_chunk(num_samples)

    def get_fsr_forces(self) -> list[float]:
        """Returns 5-finger FSR contact forces in Newtons [Thumb, Index, Middle, Ring, Little]."""
        frame = self.esp32_receiver.get_latest_frame()
        return frame.fsr_forces_n

    def get_total_grip_force_n(self) -> float:
        """Returns total contact force across all 5 fingertips."""
        frame = self.esp32_receiver.get_latest_frame()
        return frame.total_grip_force_n

    def get_latest_telemetry(self) -> ESP32TelemetryFrame:
        return self.esp32_receiver.get_latest_frame()

    def trigger_synthetic_gesture(self, gesture_name: str, duration_s: float = 1.0):
        self.synthetic_streamer.trigger_gesture(gesture_name, duration_s)
        self.esp32_receiver.trigger_synthetic_gesture(gesture_name, duration_s)

    def close(self):
        self.esp32_receiver.stop()
