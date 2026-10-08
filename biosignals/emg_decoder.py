"""
VAPA Electromyography (EMG) Muscle Signal Decoder
Decodes multi-channel forearm/arm muscle contractions into prosthetic grasp commands,
proportional grip force modulation, and emergency co-contraction abort triggers.
"""

import time
import logging
import numpy as np
from collections import deque

from config.system_config import (
    EMG_SAMPLING_RATE_HZ,
    EMG_BANDPASS_LOW_HZ,
    EMG_BANDPASS_HIGH_HZ,
    EMG_NOTCH_HZ,
    EMG_REST_THRESHOLD,
    EMG_ACTIVATION_THRESHOLD,
    EMG_HIGH_CONTRACTION_THRESHOLD,
    EMG_CO_CONTRACTION_THRESHOLD,
    RMS_WINDOW_SIZE,
    FORCE_MIN_N,
    FORCE_MAX_N,
)
from biosignals.signal_filters import SignalFilter, compute_rms_envelope, compute_mav

logger = logging.getLogger("VAPA.Biosignals.EMG")


class EMGIntent:
    """Represents decoded muscle state and intended action."""
    REST = "REST"
    GRASP_CLOSE = "GRASP_CLOSE"
    HAND_OPEN = "HAND_OPEN"
    PINCH = "PINCH"
    CO_CONTRACTION_ESTOP = "CO_CONTRACTION_ESTOP"

    def __init__(
        self,
        gesture: str,
        confidence: float,
        proportional_force_n: float,
        activation_level: float,
        channel_rms: list[float],
        timestamp: float,
    ):
        self.gesture = gesture
        self.confidence = float(confidence)
        self.proportional_force_n = float(proportional_force_n)
        self.activation_level = float(activation_level)
        self.channel_rms = channel_rms
        self.timestamp = timestamp

    def __repr__(self):
        return (
            f"EMGIntent(gesture='{self.gesture}', act={self.activation_level:.2f}, "
            f"force={self.proportional_force_n:.1f}N, conf={self.confidence:.2f})"
        )


class EMGDecoder:
    """
    Real-time multi-channel EMG processor and intent classifier.
    Channels:
      - Ch 0: Forearm Flexor (Flexor Digitorum / Grasp Close)
      - Ch 1: Forearm Extensor (Extensor Digitorum / Hand Open)
      - Ch 2: Biceps Brachii (Arm Flexion / Reach trigger)
      - Ch 3: Triceps Brachii (Arm Extension / Push)
    """
    def __init__(self, num_channels: int = 4, fs: float = EMG_SAMPLING_RATE_HZ):
        self.num_channels = num_channels
        self.fs = fs
        self.filter = SignalFilter(sampling_rate_hz=fs)
        self.filter.add_notch_filter("notch", notch_freq_hz=EMG_NOTCH_HZ)
        self.filter.add_bandpass_filter("bandpass", low_hz=EMG_BANDPASS_LOW_HZ, high_hz=EMG_BANDPASS_HIGH_HZ)

        # Baselines for calibration (noise floor / resting RMS in microvolts)
        self.baseline_rms = np.ones(num_channels) * 8.0
        self.max_rms = np.ones(num_channels) * 120.0

        # Ring buffer for continuous signal history (window of 200ms)
        self.buffer_len = int(fs * 0.20)
        self.buffers = [deque(maxlen=self.buffer_len) for _ in range(num_channels)]
        for b in self.buffers:
            b.extend([0.0] * self.buffer_len)

        # Gesture debouncing & hysteresis
        self.last_gesture = EMGIntent.REST
        self.gesture_hold_count = 0
        logger.info(f"EMGDecoder initialized ({num_channels} channels @ {fs}Hz)")

    def calibrate_baseline(self, raw_samples: np.ndarray):
        """Calibrates resting muscle baseline from quiet raw data array (channels x samples)."""
        for ch in range(min(self.num_channels, raw_samples.shape[0])):
            filtered = self.filter.filter_signal(raw_samples[ch], "notch")
            filtered = self.filter.filter_signal(filtered, "bandpass")
            rms = np.mean(compute_rms_envelope(filtered, RMS_WINDOW_SIZE))
            self.baseline_rms[ch] = max(0.005, rms)
        logger.info(f"EMG Baselines calibrated: {self.baseline_rms}")

    def update_samples(self, new_samples: np.ndarray) -> EMGIntent:
        """
        Processes new multi-channel raw samples.
        new_samples: shape (num_channels, N) or (num_channels,)
        Returns decoded EMGIntent.
        """
        now = time.time()
        if new_samples.ndim == 1:
            if len(new_samples) == self.num_channels:
                new_samples = new_samples[:, np.newaxis]
            else:
                new_samples = new_samples[np.newaxis, :]

        ch_count, n_pts = new_samples.shape
        channel_rms_list = []
        norm_activations = []

        for ch in range(min(self.num_channels, ch_count)):
            # Update ring buffer
            self.buffers[ch].extend(new_samples[ch].tolist())
            buf_arr = np.array(self.buffers[ch])

            # Apply Digital Filters
            filtered = self.filter.filter_signal(buf_arr, "notch")
            filtered = self.filter.filter_signal(filtered, "bandpass")

            # Calculate RMS envelope
            rms_vals = compute_rms_envelope(filtered, window_size=min(len(filtered), RMS_WINDOW_SIZE))
            current_rms = float(np.mean(rms_vals[-20:])) if len(rms_vals) >= 20 else float(np.mean(rms_vals))
            channel_rms_list.append(current_rms)

            # Normalize activation (0.0 to 1.0)
            base = self.baseline_rms[ch]
            max_v = max(base + 0.1, self.max_rms[ch])
            norm_act = np.clip((current_rms - base) / (max_v - base), 0.0, 1.0)
            norm_activations.append(float(norm_act))

        while len(norm_activations) < self.num_channels:
            norm_activations.append(0.0)
            channel_rms_list.append(0.0)

        flexor_act = norm_activations[0]   # Ch 0
        extensor_act = norm_activations[1] # Ch 1
        bicep_act = norm_activations[2] if self.num_channels > 2 else 0.0
        tricep_act = norm_activations[3] if self.num_channels > 3 else 0.0

        max_activation = max(flexor_act, extensor_act, bicep_act, tricep_act)

        # Proportional Grasp Force mapping: linear interpolation between FORCE_MIN_N and FORCE_MAX_N
        prop_force_n = FORCE_MIN_N + (FORCE_MAX_N - FORCE_MIN_N) * flexor_act

        # 1. Check for Co-Contraction Emergency Stop (Both flexor & extensor highly active)
        if flexor_act > EMG_CO_CONTRACTION_THRESHOLD and extensor_act > EMG_CO_CONTRACTION_THRESHOLD:
            gesture = EMGIntent.CO_CONTRACTION_ESTOP
            confidence = (flexor_act + extensor_act) / 2.0

        # 2. Check for Grasp Close Intent (Flexor dominant)
        elif flexor_act > EMG_ACTIVATION_THRESHOLD and flexor_act > extensor_act * 1.4:
            if flexor_act > EMG_HIGH_CONTRACTION_THRESHOLD:
                gesture = EMGIntent.GRASP_CLOSE
            else:
                gesture = EMGIntent.PINCH if flexor_act < 0.45 else EMGIntent.GRASP_CLOSE
            confidence = min(0.98, flexor_act / (flexor_act + extensor_act + 1e-6))

        # 3. Check for Hand Open Intent (Extensor dominant)
        elif extensor_act > EMG_ACTIVATION_THRESHOLD and extensor_act > flexor_act * 1.3:
            gesture = EMGIntent.HAND_OPEN
            confidence = min(0.98, extensor_act / (flexor_act + extensor_act + 1e-6))

        # 4. Rest
        else:
            gesture = EMGIntent.REST
            confidence = 1.0 - max_activation

        return EMGIntent(
            gesture=gesture,
            confidence=confidence,
            proportional_force_n=prop_force_n,
            activation_level=max_activation,
            channel_rms=channel_rms_list,
            timestamp=now,
        )
