"""
VAPA Electroencephalography (EEG) Brain Wave Decoder
Decodes motor imagery rhythms (Mu 8-12Hz, Beta 13-30Hz) and cognitive triggers
for target selection, reach confirmation, and mode switching.
"""

import time
import logging
import numpy as np
from collections import deque

from config.system_config import (
    EEG_SAMPLING_RATE_HZ,
    EEG_BANDPASS_LOW_HZ,
    EEG_BANDPASS_HIGH_HZ,
    EEG_NOTCH_HZ,
    EEG_MU_RHYTHM_BAND,
    EEG_BETA_RHYTHM_BAND,
    EEG_MOTOR_IMAGERY_ERD_THRESHOLD,
    EEG_ATTENTION_CONFIRM_THRESHOLD,
)
from biosignals.signal_filters import SignalFilter, compute_bandpower

logger = logging.getLogger("VAPA.Biosignals.EEG")


class EEGIntent:
    """Represents decoded brain state and cognitive command."""
    IDLE = "IDLE"
    TARGET_LOCK_CONFIRM = "TARGET_LOCK_CONFIRM"  # Cognitive confirmation to pick selected object
    INTENT_REACH = "INTENT_REACH"                # Motor imagery trigger to initiate arm trajectory
    TARGET_CYCLE_NEXT = "TARGET_CYCLE_NEXT"      # Switch to next visual object in room
    MODE_TOGGLE = "MODE_TOGGLE"                  # Toggle between autonomous and manual mode

    def __init__(
        self,
        command: str,
        confidence: float,
        mu_power: float,
        beta_power: float,
        attention_score: float,
        motor_imagery_active: bool,
        timestamp: float,
    ):
        self.command = command
        self.confidence = float(confidence)
        self.mu_power = float(mu_power)
        self.beta_power = float(beta_power)
        self.attention_score = float(attention_score)
        self.motor_imagery_active = bool(motor_imagery_active)
        self.timestamp = timestamp

    def __repr__(self):
        return (
            f"EEGIntent(cmd='{self.command}', conf={self.confidence:.2f}, "
            f"MI={self.motor_imagery_active}, Attn={self.attention_score:.2f})"
        )


class EEGDecoder:
    """
    Real-time multi-channel EEG signal processor.
    Standard electrode configuration (10-20 system):
      - Ch 0: C3 (Left Motor Cortex - Right Hand motor imagery)
      - Ch 1: C4 (Right Motor Cortex - Left Hand motor imagery)
      - Ch 2: Cz (Vertex - Sensorimotor Integration / Reach planning)
      - Ch 3: Fz / Fp1 (Frontal - Attention / Eye blink trigger)
    """
    def __init__(self, num_channels: int = 4, fs: float = EEG_SAMPLING_RATE_HZ):
        self.num_channels = num_channels
        self.fs = fs
        self.filter = SignalFilter(sampling_rate_hz=fs)
        self.filter.add_notch_filter("notch", notch_freq_hz=EEG_NOTCH_HZ)
        self.filter.add_bandpass_filter("bandpass", low_hz=EEG_BANDPASS_LOW_HZ, high_hz=EEG_BANDPASS_HIGH_HZ)

        # Baseline resting Mu power (calibrated during rest)
        self.baseline_mu_power = 0.98
        self.baseline_beta_power = 0.20

        # Ring buffers for 1.5 seconds of EEG signal
        self.buffer_len = int(fs * 1.5)
        self.buffers = [deque(maxlen=self.buffer_len) for _ in range(num_channels)]
        for b in self.buffers:
            b.extend([0.0] * self.buffer_len)

        self.last_trigger_time = 0.0
        self.cooldown_s = 0.8  # Debounce trigger
        logger.info(f"EEGDecoder initialized ({num_channels} channels @ {fs}Hz)")

    def calibrate_baseline(self, rest_samples: np.ndarray):
        """Calibrates baseline resting sensorimotor rhythm power."""
        if rest_samples.ndim > 1:
            ch_c3 = rest_samples[0]
        else:
            ch_c3 = rest_samples
        filtered = self.filter.filter_signal(ch_c3, "notch")
        filtered = self.filter.filter_signal(filtered, "bandpass")
        self.baseline_mu_power = max(0.1, compute_bandpower(filtered, self.fs, EEG_MU_RHYTHM_BAND))
        self.baseline_beta_power = max(0.05, compute_bandpower(filtered, self.fs, EEG_BETA_RHYTHM_BAND))
        logger.info(f"EEG Baseline Mu Power: {self.baseline_mu_power:.3f}, Beta: {self.baseline_beta_power:.3f}")

    def update_samples(self, new_samples: np.ndarray) -> EEGIntent:
        """
        Processes multi-channel EEG window and outputs cognitive intent.
        new_samples: shape (num_channels, N)
        """
        now = time.time()
        if new_samples.ndim == 1:
            if len(new_samples) == self.num_channels:
                new_samples = new_samples[:, np.newaxis]
            else:
                new_samples = new_samples[np.newaxis, :]

        ch_count, _ = new_samples.shape
        for ch in range(min(self.num_channels, ch_count)):
            self.buffers[ch].extend(new_samples[ch].tolist())

        # Analyze Motor Cortex channel (C3 or Cz)
        c3_signal = np.array(self.buffers[0])
        c3_filtered = self.filter.filter_signal(c3_signal, "notch")
        c3_filtered = self.filter.filter_signal(c3_filtered, "bandpass")

        mu_power = compute_bandpower(c3_filtered, self.fs, EEG_MU_RHYTHM_BAND)
        beta_power = compute_bandpower(c3_filtered, self.fs, EEG_BETA_RHYTHM_BAND)

        # Analyze Frontal channel (Fz) for attention & artifact triggers
        fz_idx = min(3, self.num_channels - 1)
        fz_signal = np.array(self.buffers[fz_idx])
        fz_filtered = self.filter.filter_signal(fz_signal, "notch")
        fz_filtered = self.filter.filter_signal(fz_filtered, "bandpass")
        
        # High frontal amplitude spike corresponds to intentional double blink or attention burst
        frontal_peak = float(np.max(np.abs(fz_filtered[-int(self.fs * 0.3):])))
        
        # Attention score: ratio of Beta (active focus) to Mu/Theta
        attention_score = float(np.clip(beta_power / (mu_power + 1e-6) * 0.7, 0.0, 1.0))

        # Event-Related Desynchronization (ERD) calculation:
        # Motor imagery causes power decrease in Mu band compared to resting baseline
        erd = (self.baseline_mu_power - mu_power) / (self.baseline_mu_power + 1e-6)
        motor_imagery_active = erd > EEG_MOTOR_IMAGERY_ERD_THRESHOLD

        command = EEGIntent.IDLE
        confidence = 0.5

        # Debounce trigger
        if now - self.last_trigger_time > self.cooldown_s:
            # 1. Frontal spike trigger -> cycle target or mode switch
            if frontal_peak > 65.0:  # Microvolt threshold
                command = EEGIntent.TARGET_CYCLE_NEXT
                confidence = 0.92
                self.last_trigger_time = now

            # 2. Strong Motor Imagery (ERD) -> Trigger Reach
            elif motor_imagery_active and beta_power > 0.15:
                command = EEGIntent.INTENT_REACH
                confidence = min(0.95, 0.60 + erd * 0.4)
                self.last_trigger_time = now

            # 3. High Attention focus -> Confirm Target Lock
            elif attention_score > EEG_ATTENTION_CONFIRM_THRESHOLD:
                command = EEGIntent.TARGET_LOCK_CONFIRM
                confidence = attention_score
                self.last_trigger_time = now

        return EEGIntent(
            command=command,
            confidence=confidence,
            mu_power=mu_power,
            beta_power=beta_power,
            attention_score=attention_score,
            motor_imagery_active=motor_imagery_active,
            timestamp=now,
        )
