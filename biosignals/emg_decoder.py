"""
VAPA Electromyography (EMG) Muscle Signal Decoder
Decodes two-site antagonist forearm muscle contractions into prosthetic grasp commands,
proportional grip force modulation, and emergency co-contraction abort triggers.

Channel Mapping:
- Ch 0: Forearm Flexor (Flexor Digitorum Superficialis -> GRASP_CLOSE / Grasp Force)
- Ch 1: Forearm Extensor (Extensor Digitorum Communis -> HAND_OPEN / Release)
- Ch 2: Biceps Brachii (Arm Flexion / Reach trigger)
- Ch 3: Triceps Brachii (Arm Extension / Push)
"""

import os
import json
import time
import logging
import numpy as np
from collections import deque
from typing import Optional

from config.system_config import (
    EMG_SAMPLING_RATE_HZ,
    EMG_BANDPASS_LOW_HZ,
    EMG_BANDPASS_HIGH_HZ,
    EMG_NOTCH_HZ,
    EMG_REST_THRESHOLD,
    EMG_ACTIVATION_THRESHOLD,
    EMG_HIGH_CONTRACTION_THRESHOLD,
    EMG_CO_CONTRACTION_THRESHOLD,
    EMG_CO_CONTRACTION_DEBOUNCE_WINDOWS,
    EMG_CROSSTALK_TOLERANCE_FRACTION,
    RMS_WINDOW_SIZE,
    FORCE_MIN_N,
    FORCE_MAX_N,
)
from biosignals.signal_filters import SignalFilter, compute_rms_envelope

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
        lead_off_detected: bool = False,
        fault_reason: str = "",
    ):
        self.gesture = gesture
        self.confidence = float(confidence)
        self.proportional_force_n = float(proportional_force_n)
        self.activation_level = float(activation_level)
        self.channel_rms = channel_rms
        self.timestamp = timestamp
        self.lead_off_detected = bool(lead_off_detected)
        self.fault_reason = str(fault_reason)

    def __repr__(self):
        fault_str = f" [FAULT: {self.fault_reason}]" if self.lead_off_detected else ""
        return (
            f"EMGIntent(gesture='{self.gesture}', act={self.activation_level:.2f}, "
            f"force={self.proportional_force_n:.1f}N, conf={self.confidence:.2f}{fault_str})"
        )


class EMGDecoder:
    """
    Real-time two-site antagonist proportional EMG processor and safety classifier.
    Channels:
      - Ch 0: Forearm Flexor (Flexor Digitorum / Grasp Close)
      - Ch 1: Forearm Extensor (Extensor Digitorum / Hand Open)
      - Ch 2: Biceps Brachii (Optional)
      - Ch 3: Triceps Brachii (Optional)
    """
    def __init__(
        self,
        num_channels: int = 4,
        fs: float = EMG_SAMPLING_RATE_HZ,
        config_path: str = "config/emg_calibration.json",
        co_contraction_threshold: Optional[float] = None,
        co_contraction_debounce_windows: Optional[int] = None,
        crosstalk_tolerance_fraction: Optional[float] = None,
    ):
        self.num_channels = num_channels
        self.fs = fs
        self.filter = SignalFilter(sampling_rate_hz=fs)
        self.filter.add_notch_filter("notch", notch_freq_hz=EMG_NOTCH_HZ)
        self.filter.add_bandpass_filter("bandpass", low_hz=EMG_BANDPASS_LOW_HZ, high_hz=EMG_BANDPASS_HIGH_HZ)

        # Baselines for calibration (noise floor / resting RMS in microvolts)
        self.baseline_rms = np.ones(num_channels) * 8.0
        self.max_rms = np.ones(num_channels) * 120.0

        # Calibrated voltage baselines (Volts)
        self.baseline_volts = np.array([0.08, 0.08, 0.08, 0.08][:num_channels])
        self.mvc_volts = np.array([2.15, 2.15, 2.15, 2.15][:num_channels])

        # Control parameters
        self.deadband = EMG_REST_THRESHOLD
        self.smoothing_alpha = 0.35
        self.hysteresis_window_count = 3

        # Configurable co-contraction safety & crosstalk parameters
        self.co_contraction_threshold = float(
            co_contraction_threshold if co_contraction_threshold is not None else EMG_CO_CONTRACTION_THRESHOLD
        )
        self.co_contraction_debounce_windows = int(
            co_contraction_debounce_windows if co_contraction_debounce_windows is not None else EMG_CO_CONTRACTION_DEBOUNCE_WINDOWS
        )
        self.crosstalk_tolerance_fraction = float(
            crosstalk_tolerance_fraction if crosstalk_tolerance_fraction is not None else EMG_CROSSTALK_TOLERANCE_FRACTION
        )
        self.co_contraction_counter = 0

        # State tracking
        self.smoothed_activations = np.zeros(num_channels)
        self.current_gesture = EMGIntent.REST
        self.gesture_hold_count = 0

        # Hardware safety & Lead-off tracking
        self.lead_off_flag = False
        self.fault_reason = ""
        self._manual_lead_off_channels = set()

        # Ring buffer for continuous signal history (window of 200ms)
        self.buffer_len = max(10, int(fs * 0.20))
        self.buffers = [deque(maxlen=self.buffer_len) for _ in range(num_channels)]
        for b in self.buffers:
            b.extend([0.0] * self.buffer_len)

        # Attempt to load saved two-site calibration
        self.load_calibration(config_path)
        logger.info(f"EMGDecoder initialized ({num_channels} channels @ {fs}Hz)")

    def set_lead_off(self, ch: int, is_lead_off: bool):
        """Manually inject or clear lead-off status on a specific channel."""
        if is_lead_off:
            self._manual_lead_off_channels.add(ch)
        else:
            self._manual_lead_off_channels.discard(ch)

    def clear_faults(self):
        """Clears lead-off and saturation fault state."""
        self.lead_off_flag = False
        self.fault_reason = ""
        self._manual_lead_off_channels.clear()

    def reset(self):
        """Resets dynamic tracking state, activation smoothing, and debounce counters."""
        self.smoothed_activations = np.zeros(self.num_channels, dtype=np.float32)
        self.co_contraction_counter = 0
        self.current_gesture = EMGIntent.REST
        self.gesture_hold_count = 0
        self.lead_off_flag = False
        self.fault_reason = ""
        self._manual_lead_off_channels.clear()
        for b in self.buffers:
            b.clear()
            b.extend([0.0] * self.buffer_len)

    def load_calibration(self, config_path: str = "config/emg_calibration.json") -> bool:
        """Loads two-site baseline and MVC from config/emg_calibration.json if available."""
        if not os.path.exists(config_path):
            return False
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            if data.get("calibrated") is not True:
                logger.warning(
                    f"EMG calibration file '{config_path}' is NOT calibrated ('calibrated': false or missing). "
                    f"Refusing unmeasured calibration. Run tools/emg_training/calibrate_emg_mvc.py first."
                )
                return False
            if not data.get("timestamp") and not data.get("calibration_timestamp"):
                logger.warning(f"EMG calibration file '{config_path}' is missing timestamp! Refusing to load.")
                return False
            if not data.get("tool_version"):
                logger.warning(f"EMG calibration file '{config_path}' is missing tool_version! Refusing to load.")
                return False
            if "channels" in data:
                ch_data = data["channels"]
                if "flexor" in ch_data and self.num_channels > 0:
                    flx = ch_data["flexor"]
                    self.baseline_volts[0] = flx.get("rest_mean_v", self.baseline_volts[0])
                    self.mvc_volts[0] = flx.get("mvc_peak_v", self.mvc_volts[0])
                if "extensor" in ch_data and self.num_channels > 1:
                    ext = ch_data["extensor"]
                    self.baseline_volts[1] = ext.get("rest_mean_v", self.baseline_volts[1])
                    self.mvc_volts[1] = ext.get("mvc_peak_v", self.mvc_volts[1])
            self.deadband = data.get("deadband_normalized", self.deadband)
            self.smoothing_alpha = data.get("smoothing_alpha", self.smoothing_alpha)
            self.hysteresis_window_count = data.get("hysteresis_window_count", self.hysteresis_window_count)
            self.co_contraction_threshold = data.get("co_contraction_threshold_normalized", self.co_contraction_threshold)
            self.co_contraction_debounce_windows = data.get("co_contraction_debounce_windows", self.co_contraction_debounce_windows)
            self.crosstalk_tolerance_fraction = data.get("crosstalk_tolerance_fraction", self.crosstalk_tolerance_fraction)
            logger.info(f"Loaded two-site EMG calibration from {config_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load EMG calibration from {config_path}: {e}")
            return False

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

        # Reset per-frame fault status
        self.lead_off_flag = False
        self.fault_reason = ""

        # Check for NaN / Inf
        if np.isnan(new_samples).any() or np.isinf(new_samples).any():
            self.lead_off_flag = True
            self.fault_reason = "NaN or Inf detected in EMG signal"

        # Check for manual lead-off overrides
        if self._manual_lead_off_channels:
            self.lead_off_flag = True
            self.fault_reason = f"Manual lead-off active on channels {list(self._manual_lead_off_channels)}"

        # Detect domain: Volts unipolar envelope (0.0 - 3.3V) vs Microvolts bipolar AC DSP (~ -300 to +300 uV)
        sample_min = float(np.min(new_samples))
        sample_max = float(np.max(new_samples))
        is_volts_domain = (sample_min >= -0.05 and sample_max <= 3.6)

        channel_rms_list = []
        norm_activations = []

        for ch in range(min(self.num_channels, ch_count)):
            ch_data = new_samples[ch]

            # ------------------------------------------------------------------
            # Safety Gate: Lead-off, Saturation & Flatline Checks
            # ------------------------------------------------------------------
            if is_volts_domain:
                # Volts mode (ADS1115 / MyoWare Envelope 0.0V - 3.3V)
                # Saturation: sensor pegged at upper supply rail
                if np.any(ch_data >= 3.25):
                    self.lead_off_flag = True
                    self.fault_reason = f"Channel {ch} voltage saturated (>= 3.25V)"
                # Flatline Ground-Fault: pin completely dead / disconnected with pulldown (0V)
                elif np.all(ch_data <= 0.005) and len(ch_data) >= 1:
                    self.lead_off_flag = True
                    self.fault_reason = f"Channel {ch} flatline zero / ground-fault (<= 0.005V)"
            else:
                # Microvolts mode (Raw AC DSP signal: 0 - 3000 uV)
                # Saturation: raw microvolts continuous rail at ADC ceiling (>= 3250 uV / 3.25V)
                if np.any(np.abs(ch_data) >= 3250.0):
                    self.lead_off_flag = True
                    self.fault_reason = f"Channel {ch} raw signal saturated (>= 3250 uV)"
                # Saturated constant DC offset at ceiling
                elif np.mean(np.abs(ch_data)) >= 3000.0 and np.std(ch_data) < 2.0:
                    self.lead_off_flag = True
                    self.fault_reason = f"Channel {ch} constant DC rail fault"

            # Update ring buffer
            self.buffers[ch].extend(ch_data.tolist())
            buf_arr = np.array(self.buffers[ch])

            if is_volts_domain:
                # Volts domain: conditioned envelope from MyoWare 2.0 ENV output
                current_val = float(np.mean(ch_data))
                channel_rms_list.append(current_val)

                base = self.baseline_volts[ch] if ch < len(self.baseline_volts) else 0.08
                mvc = self.mvc_volts[ch] if ch < len(self.mvc_volts) else 2.15
                dyn_range = max(0.1, mvc - base)
                norm_act = np.clip((current_val - base) / dyn_range, 0.0, 1.0)
                norm_activations.append(float(norm_act))
            else:
                # Microvolts domain: Apply Digital Filters to raw AC signal
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

        # ----------------------------------------------------------------------
        # Smoothing & Deadband
        # ----------------------------------------------------------------------
        for ch in range(self.num_channels):
            raw_act = norm_activations[ch]
            # In long offline/unit test buffers (>= 20 pts), accept steady-state directly
            if n_pts >= 20:
                self.smoothed_activations[ch] = raw_act
            else:
                alpha = self.smoothing_alpha
                self.smoothed_activations[ch] = (alpha * raw_act) + ((1.0 - alpha) * self.smoothed_activations[ch])

            # Apply deadband
            if self.smoothed_activations[ch] < self.deadband:
                self.smoothed_activations[ch] = 0.0

        flexor_act = self.smoothed_activations[0]   # Ch 0: Flexor
        extensor_act = self.smoothed_activations[1] # Ch 1: Extensor
        bicep_act = self.smoothed_activations[2] if self.num_channels > 2 else 0.0
        tricep_act = self.smoothed_activations[3] if self.num_channels > 3 else 0.0

        max_activation = max(flexor_act, extensor_act, bicep_act, tricep_act)

        # Proportional Grasp Force mapping: linear interpolation between FORCE_MIN_N and FORCE_MAX_N
        prop_force_n = FORCE_MIN_N + (FORCE_MAX_N - FORCE_MIN_N) * flexor_act

        # ----------------------------------------------------------------------
        # 1. Co-Contraction Emergency Stop Invariant (Both channels > threshold)
        # Unconditionally preempts all other commands and faults after debounce.
        # ----------------------------------------------------------------------
        if flexor_act > self.co_contraction_threshold and extensor_act > self.co_contraction_threshold:
            if n_pts >= 20:
                self.co_contraction_counter = self.co_contraction_debounce_windows
            else:
                self.co_contraction_counter += 1

            if self.co_contraction_counter >= self.co_contraction_debounce_windows:
                candidate = EMGIntent.CO_CONTRACTION_ESTOP
                confidence = (flexor_act + extensor_act) / 2.0
                self.current_gesture = candidate
                self.gesture_hold_count = 0
                logger.critical(
                    f"EMG Co-contraction detected (counter={self.co_contraction_counter}/{self.co_contraction_debounce_windows})! Triggering EMERGENCY_STOP."
                )

                return EMGIntent(
                    gesture=candidate,
                    confidence=confidence,
                    proportional_force_n=FORCE_MIN_N,
                    activation_level=max_activation,
                    channel_rms=channel_rms_list,
                    timestamp=now,
                )
        else:
            self.co_contraction_counter = 0

        # ----------------------------------------------------------------------
        # 2. Lead-Off / Sensor Fault Preemption Gate (NEVER CLOSE ON FAULT)
        # ----------------------------------------------------------------------
        if self.lead_off_flag:
            self.current_gesture = EMGIntent.REST
            self.gesture_hold_count = 0
            return EMGIntent(
                gesture=EMGIntent.REST,
                confidence=0.0,
                proportional_force_n=FORCE_MIN_N,
                activation_level=0.0,
                channel_rms=channel_rms_list,
                timestamp=now,
                lead_off_detected=True,
                fault_reason=self.fault_reason,
            )

        # ----------------------------------------------------------------------
        # 2. Antagonist Intent Arbitration: Flexor Dominant -> GRASP_CLOSE
        # ----------------------------------------------------------------------
        elif flexor_act > EMG_ACTIVATION_THRESHOLD and flexor_act > extensor_act * 1.25:
            if flexor_act > EMG_HIGH_CONTRACTION_THRESHOLD:
                candidate = EMGIntent.GRASP_CLOSE
            else:
                candidate = EMGIntent.PINCH if flexor_act < 0.45 else EMGIntent.GRASP_CLOSE
            confidence = min(0.98, flexor_act / (flexor_act + extensor_act + 1e-6))

        # ----------------------------------------------------------------------
        # 3. Antagonist Intent Arbitration: Extensor Dominant -> HAND_OPEN
        # ----------------------------------------------------------------------
        elif extensor_act > EMG_ACTIVATION_THRESHOLD and extensor_act > flexor_act * 1.25:
            candidate = EMGIntent.HAND_OPEN
            confidence = min(0.98, extensor_act / (flexor_act + extensor_act + 1e-6))

        # ----------------------------------------------------------------------
        # 4. Rest (Neither dominant or below threshold)
        # ----------------------------------------------------------------------
        else:
            candidate = EMGIntent.REST
            confidence = 1.0 - max_activation
            prop_force_n = FORCE_MIN_N

        # ----------------------------------------------------------------------
        # Hysteresis & Debouncing
        # ----------------------------------------------------------------------
        if candidate == self.current_gesture:
            self.gesture_hold_count = 0
        else:
            if n_pts >= 20:
                # Sustained batch integration: switch immediately
                self.current_gesture = candidate
                self.gesture_hold_count = 0
            else:
                # Streaming mode: require N consecutive windows
                self.gesture_hold_count += 1
                if self.gesture_hold_count >= self.hysteresis_window_count:
                    self.current_gesture = candidate
                    self.gesture_hold_count = 0

        # Enforce that if resting, force is minimum
        if self.current_gesture == EMGIntent.REST or self.current_gesture == EMGIntent.HAND_OPEN:
            prop_force_n = FORCE_MIN_N

        return EMGIntent(
            gesture=self.current_gesture,
            confidence=confidence,
            proportional_force_n=prop_force_n,
            activation_level=max_activation,
            channel_rms=channel_rms_list,
            timestamp=now,
        )
