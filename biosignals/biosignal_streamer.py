"""
VAPA Biosignal Streamer Interface
Connects to real serial DAQ hardware (Arduino/Teensy/ADS1299/OpenBCI)
or provides a realistic synthetic bio-signal generator for offline testing.
"""

import time
import math
import logging
import threading
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from config.hardware_config import (
    BIOSIGNAL_SERIAL_PORT,
    BIOSIGNAL_BAUD_RATE,
    BIOSIGNAL_TIMEOUT_S,
    NUM_EMG_CHANNELS,
    NUM_EEG_CHANNELS,
)
from config.system_config import EMG_SAMPLING_RATE_HZ, EEG_SAMPLING_RATE_HZ

logger = logging.getLogger("VAPA.Biosignals.Streamer")


class SyntheticBiosignalStreamer:
    """
    Generates realistic physiological multi-channel EMG and EEG waveforms.
    Allows injecting synthetic muscle contractions and motor imagery for testing.
    """
    def __init__(self, num_emg=NUM_EMG_CHANNELS, num_eeg=NUM_EEG_CHANNELS):
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
        logger.info("SyntheticBiosignalStreamer initialized.")

    def trigger_gesture(self, gesture_name: str, duration_s: float = 1.0):
        """Helper to inject simulated user gesture."""
        def _reset_after(prop_name, d):
            time.sleep(d)
            setattr(self, prop_name, False)

        gesture_map = {
            "grasp": "inject_emg_grasp",
            "open": "inject_emg_open",
            "estop": "inject_emg_estop",
            "reach": "inject_eeg_reach",
            "cycle": "inject_eeg_cycle",
        }
        if gesture_name in gesture_map:
            attr = gesture_map[gesture_name]
            setattr(self, attr, True)
            t = threading.Thread(target=_reset_after, args=(attr, duration_s), daemon=True)
            t.start()
            logger.info(f"Triggered synthetic gesture: {gesture_name} for {duration_s}s")

    def read_chunk(self, num_samples: int = 20):
        """
        Generates a chunk of EMG samples (shape: num_emg x num_samples)
        and EEG samples (shape: num_eeg x num_samples).
        """
        t_arr = np.linspace(self.phase, self.phase + num_samples / EMG_SAMPLING_RATE_HZ, num_samples)
        self.phase += num_samples / EMG_SAMPLING_RATE_HZ

        # 1. Base EMG (Raw microvolts: White noise + 50Hz hum + baseline muscle tone)
        emg_data = np.random.normal(0, 3.0, (self.num_emg, num_samples))
        emg_data += 2.0 * np.sin(2 * np.pi * 50.0 * t_arr)  # 50Hz line hum

        # Muscle Contraction Modulation:
        # Ch 0: Flexor, Ch 1: Extensor
        if self.inject_emg_grasp:
            # Strong burst in flexor channel (burst of 100-350Hz high amplitude)
            burst = np.random.normal(0, 120.0, num_samples) * (1.0 + 0.5 * np.sin(2 * np.pi * 150 * t_arr))
            emg_data[0] += burst
        elif self.inject_emg_open:
            # Strong burst in extensor channel
            burst = np.random.normal(0, 110.0, num_samples) * (1.0 + 0.5 * np.sin(2 * np.pi * 180 * t_arr))
            emg_data[1] += burst
        elif self.inject_emg_estop:
            # Co-contraction in both channels
            burst = np.random.normal(0, 150.0, num_samples)
            emg_data[0] += burst
            emg_data[1] += burst

        # 2. Base EEG (Ch 0: C3, Ch 1: C4, Ch 2: Cz, Ch 3: Fz)
        eeg_data = np.random.normal(0, 4.0, (self.num_eeg, num_samples))
        # Add 10Hz Mu rhythm to motor cortex
        if not self.inject_eeg_reach:
            # Resting Mu rhythm present (high power at rest)
            eeg_data[0] += 25.0 * np.sin(2 * np.pi * 10.0 * t_arr)
            eeg_data[2] += 20.0 * np.sin(2 * np.pi * 10.0 * t_arr)
        else:
            # Motor imagery ERD (Mu rhythm suppresses, Beta 22Hz increases)
            eeg_data[0] += 4.0 * np.sin(2 * np.pi * 10.0 * t_arr)
            eeg_data[0] += 18.0 * np.sin(2 * np.pi * 22.0 * t_arr)

        if self.inject_eeg_cycle:
            # Frontal artifact spike (eye blink / attention trigger)
            eeg_data[3] += 80.0

        return emg_data, eeg_data


class BiosignalStreamer:
    """
    Hardware Serial Biosignal Streamer with auto-fallback to SyntheticStreamer.
    Parses comma-separated or binary frames from microcontrollers:
    Format: "EMG0,EMG1,EMG2,EMG3,EEG0,EEG1,EEG2,EEG3\n"
    """
    def __init__(self, port=BIOSIGNAL_SERIAL_PORT, baud_rate=BIOSIGNAL_BAUD_RATE, force_mock=False):
        self.port = port
        self.baud_rate = baud_rate
        self.is_synthetic = False
        self.serial_conn = None
        self.running = False

        if force_mock or not SERIAL_AVAILABLE:
            self._init_synthetic()
            return

        try:
            self._init_hardware()
        except Exception as e:
            logger.warning(f"Could not open serial port {port} ({e}). Using SyntheticBiosignalStreamer.")
            self._init_synthetic()

    def _init_synthetic(self):
        self.is_synthetic = True
        self.synthetic_streamer = SyntheticBiosignalStreamer()

    def _init_hardware(self):
        self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=BIOSIGNAL_TIMEOUT_S)
        logger.info(f"Connected to hardware biosignal DAQ on {self.port} @ {self.baud_rate} baud.")

    def read_chunk(self, num_samples: int = 20):
        """Reads latest chunk of multi-channel data."""
        if self.is_synthetic:
            return self.synthetic_streamer.read_chunk(num_samples)

        try:
            emg_list = [[] for _ in range(NUM_EMG_CHANNELS)]
            eeg_list = [[] for _ in range(NUM_EEG_CHANNELS)]
            count = 0

            while self.serial_conn.in_waiting > 0 and count < num_samples:
                line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= NUM_EMG_CHANNELS + NUM_EEG_CHANNELS:
                    for i in range(NUM_EMG_CHANNELS):
                        emg_list[i].append(float(parts[i]))
                    for j in range(NUM_EEG_CHANNELS):
                        eeg_list[j].append(float(parts[NUM_EMG_CHANNELS + j]))
                    count += 1

            if count > 0:
                return np.array(emg_list), np.array(eeg_list)
            else:
                return np.zeros((NUM_EMG_CHANNELS, num_samples)), np.zeros((NUM_EEG_CHANNELS, num_samples))
        except Exception as e:
            logger.error(f"Error reading biosignal serial stream: {e}")
            return np.zeros((NUM_EMG_CHANNELS, num_samples)), np.zeros((NUM_EEG_CHANNELS, num_samples))

    def trigger_synthetic_gesture(self, gesture_name: str, duration_s: float = 1.0):
        if self.is_synthetic:
            self.synthetic_streamer.trigger_gesture(gesture_name, duration_s)

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Biosignal serial stream closed.")
