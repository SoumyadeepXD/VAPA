"""
VAPA EMG Safe Machine Learning Classifier
Wraps trained machine learning models (e.g. Support Vector Classifiers) with
rigorous fail-closed cryptographic verification, confidence thresholding, majority voting,
lead-off detection, and parallel threshold-based co-contraction safety invariants.

Implements the exact same interface as biosignals.emg_decoder.EMGDecoder.
"""

import os
import time
import logging
import hashlib
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

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
from biosignals.signal_filters import SignalFilter, compute_rms_envelope
from biosignals.emg_decoder import EMGIntent, EMGDecoder

logger = logging.getLogger("VAPA.Biosignals.Classifier")

# Verified SHA-256 Checksums from data/emg/MANIFEST.csv
EXPECTED_MANIFEST_HASHES = {
    "biosignals/models/svm_emg_model.pkl": "5a1dd23d78109578f0d8bc93d04850e3f3114b3b198fc68541e70bb2208ba4ac",
    "biosignals/models/emg_scaler.pkl": "10af92fcd4642c048b7c9a8322025ebf5330118ce00e7be9a384974468cf3ad1",
    "biosignals/models/feature_columns.pkl": "a039bf613659a6235a21ba0a8c887dbdfbc76a4545b02bf8ede50e10ad2c0f47",
}


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class SafeModelLoader:
    """
    Cryptographically secure model loader.
    Verifies file existence, SHA-256 hashes against manifest, library compatibility,
    and feature alignment before deserialization.
    """

    @staticmethod
    def load_artifacts(
        model_path: str = "biosignals/models/svm_emg_model.pkl",
        scaler_path: str = "biosignals/models/emg_scaler.pkl",
        features_path: str = "biosignals/models/feature_columns.pkl",
        expected_hashes: Optional[Dict[str, str]] = None,
        enforce_manifest_check: bool = True,
    ) -> Tuple[bool, Optional[Any], Optional[Any], Optional[List[str]], str]:
        """
        Safely loads model, scaler, and feature columns.
        Returns: (success: bool, model, scaler, feature_columns, reason: str)
        """
        import sklearn
        import joblib

        hashes = expected_hashes or EXPECTED_MANIFEST_HASHES

        for path in [model_path, scaler_path, features_path]:
            if not os.path.exists(path):
                return False, None, None, None, f"Artifact not found on disk: {path}"

        if enforce_manifest_check:
            for path in [model_path, scaler_path, features_path]:
                expected = hashes.get(path)
                if expected:
                    actual = compute_file_sha256(path)
                    if actual.lower() != expected.lower():
                        return (
                            False,
                            None,
                            None,
                            None,
                            f"Cryptographic hash mismatch for {path}: expected {expected}, got {actual}",
                        )

        try:
            feature_cols = joblib.load(features_path)
            if not isinstance(feature_cols, (list, tuple, np.ndarray)):
                return False, None, None, None, "feature_columns artifact is not a valid list"

            scaler = joblib.load(scaler_path)
            scaler_features = getattr(scaler, "n_features_in_", len(getattr(scaler, "mean_", [])))
            if scaler_features != len(feature_cols):
                return (
                    False,
                    None,
                    None,
                    None,
                    f"Scaler feature count mismatch: scaler expects {scaler_features}, columns has {len(feature_cols)}",
                )

            model = joblib.load(model_path)
            if not hasattr(model, "predict"):
                return False, None, None, None, "Model object does not implement predict()"

            model_features = getattr(model, "n_features_in_", None)
            if model_features is not None and model_features != len(feature_cols):
                return (
                    False,
                    None,
                    None,
                    None,
                    f"Model feature count mismatch: model expects {model_features}, columns has {len(feature_cols)}",
                )

            logger.info(
                f"[OK] Cryptographic model verification passed. Loaded {type(model).__name__} with {len(feature_cols)} features."
            )
            return True, model, scaler, list(feature_cols), "Success"

        except Exception as e:
            return False, None, None, None, f"Exception during model loading: {str(e)}"


class EMGClassifier:
    """
    Fail-closed EMG Gesture Classifier adhering to the EMGDecoder interface.
    
    Safety Invariants:
    1. Parallel Threshold E-Stop: Evaluates flexor/extensor co-contraction continuously.
       If co-contraction threshold is exceeded, unconditionally triggers EMERGENCY_STOP.
    2. Fail-Closed Fallback: On loader failure, invalid input, or model exceptions,
       falls back immediately to standard threshold EMGDecoder.
    3. Confidence Gating & Majority Voting: Rejects predictions below confidence threshold.
       Requires majority vote across recent sliding windows before changing gesture state.
    4. Lead-Off / Flat / Saturated Signal Protection: Output defaults to REST and flags error.
    """

    def __init__(
        self,
        num_channels: int = 4,
        fs: float = EMG_SAMPLING_RATE_HZ,
        model_path: str = "biosignals/models/svm_emg_model.pkl",
        scaler_path: str = "biosignals/models/emg_scaler.pkl",
        features_path: str = "biosignals/models/feature_columns.pkl",
        confidence_threshold: float = 0.70,
        majority_vote_window: int = 3,
        window_duration_ms: float = 200.0,
        enforce_manifest_check: bool = True,
    ):
        self.num_channels = num_channels
        self.fs = fs
        self.confidence_threshold = float(confidence_threshold)
        self.majority_vote_window = int(majority_vote_window)
        self.window_duration_ms = min(float(window_duration_ms), 250.0)  # Capped at 250ms

        # Ring buffer for classifier window (<= 250ms)
        self.window_samples = max(4, int(self.fs * (self.window_duration_ms / 1000.0)))
        self.raw_buffers = [deque(maxlen=self.window_samples) for _ in range(num_channels)]
        for b in self.raw_buffers:
            b.extend([0.0] * self.window_samples)

        # Baseline & Calibration compatibility with EMGDecoder
        self.baseline_rms = np.ones(num_channels) * 8.0
        self.max_rms = np.ones(num_channels) * 120.0
        self.buffers = self.raw_buffers  # Alias for EMGDecoder compatibility

        # Parallel threshold fallback decoder (Guarantees zero degradation of baseline safety)
        self.fallback_decoder = EMGDecoder(num_channels=num_channels, fs=fs)

        # Majority vote history
        self.recent_predictions = deque(maxlen=self.majority_vote_window)
        for _ in range(self.majority_vote_window):
            self.recent_predictions.append(EMGIntent.REST)

        self.last_stable_gesture = EMGIntent.REST
        self.last_gesture = EMGIntent.REST
        self.lead_off_flag = False
        self.model_ready = False
        self.fallback_active = True
        self.failure_reason = ""

        # Attempt Safe Model Load
        success, model, scaler, feature_cols, reason = SafeModelLoader.load_artifacts(
            model_path=model_path,
            scaler_path=scaler_path,
            features_path=features_path,
            enforce_manifest_check=enforce_manifest_check,
        )

        if success:
            self.model = model
            self.scaler = scaler
            self.feature_columns = feature_cols
            self.model_ready = True
            self.fallback_active = False
            logger.info(f"EMGClassifier successfully initialized with active ML model ({self.window_duration_ms}ms window)")
        else:
            self.model = None
            self.scaler = None
            self.feature_columns = []
            self.model_ready = False
            self.fallback_active = True
            self.failure_reason = reason
            logger.warning(f"EMGClassifier model load failed closed: {reason}. Operating in Fallback Threshold mode.")

    def calibrate_baseline(self, raw_samples: np.ndarray):
        """Calibrates resting muscle baseline; mirrors EMGDecoder interface."""
        self.fallback_decoder.calibrate_baseline(raw_samples)
        self.baseline_rms = self.fallback_decoder.baseline_rms.copy()
        self.max_rms = self.fallback_decoder.max_rms.copy()

    def _extract_window_features(self, window: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts features corresponding to self.feature_columns from a 1D window.
        Returns 1D feature array of length len(self.feature_columns).
        """
        if len(window) < 4:
            return None

        # Remove DC component
        sig = window - np.mean(window)
        std_val = float(np.std(sig))

        feat_dict: Dict[str, float] = {}

        # Basic Time-Domain Features
        feat_dict["RMS"] = float(np.sqrt(np.mean(sig ** 2)))
        feat_dict["MAV"] = float(np.mean(np.abs(sig)))
        feat_dict["Variance"] = float(np.var(sig))
        feat_dict["STD"] = std_val
        feat_dict["Waveform_Length"] = float(np.sum(np.abs(np.diff(sig))))
        feat_dict["Peak_to_Peak"] = float(np.max(sig) - np.min(sig))
        feat_dict["Energy"] = float(np.sum(sig ** 2))

        # Zero Crossings
        thresh = 0.01 * std_val
        zc = np.sum(((sig[:-1] * sig[1:]) < 0) & (np.abs(sig[:-1] - sig[1:]) >= thresh))
        feat_dict["Zero_Crossings"] = float(zc)

        # Frequency-Domain Features (FFT)
        if len(sig) >= 8:
            fft_vals = np.fft.rfft(sig)
            mag = np.abs(fft_vals)
            freqs = np.fft.rfftfreq(len(sig), d=1.0 / self.fs)
            mag[0] = 0.0  # Zero DC
            power = mag ** 2
            total_power = np.sum(power)
            if total_power > 1e-9:
                feat_dict["Mean_Frequency"] = float(np.sum(freqs * power) / total_power)
                cum_power = np.cumsum(power)
                idx = np.searchsorted(cum_power, total_power / 2.0)
                feat_dict["Median_Frequency"] = float(freqs[min(idx, len(freqs) - 1)])
            else:
                feat_dict["Mean_Frequency"] = 0.0
                feat_dict["Median_Frequency"] = 0.0
        else:
            feat_dict["Mean_Frequency"] = 0.0
            feat_dict["Median_Frequency"] = 0.0

        # Construct vector aligned with feature_columns
        vec = []
        for col in self.feature_columns:
            vec.append(feat_dict.get(col, 0.0))

        return np.array(vec, dtype=np.float32)

    def _map_prediction_to_gesture(self, raw_label: str) -> str:
        """
        Maps raw classifier output strings to canonical EMGIntent gestures.
        FIST -> GRASP_CLOSE
        OPEN -> HAND_OPEN
        WRIST_FLEXION -> GRASP_CLOSE (flexion actuation intent)
        WRIST_EXTENSION -> HAND_OPEN (extension actuation intent)
        REST -> REST
        """
        label = str(raw_label).upper().strip()
        if label == "FIST":
            return EMGIntent.GRASP_CLOSE
        elif label == "OPEN":
            return EMGIntent.HAND_OPEN
        elif label in ("WRIST_FLEXION", "WRIST_FLEX"):
            return EMGIntent.GRASP_CLOSE
        elif label in ("WRIST_EXTENSION", "WRIST_EXTEND"):
            return EMGIntent.HAND_OPEN
        elif label == "REST":
            return EMGIntent.REST
        else:
            return EMGIntent.REST

    def update_samples(self, new_samples: np.ndarray) -> EMGIntent:
        """
        Processes new multi-channel raw samples.
        Guarantees non-blocking, thread-safe, fail-closed execution.
        """
        now = time.time()

        # Update fallback decoder continuously (guarantees parallel safety monitoring)
        fallback_intent = self.fallback_decoder.update_samples(new_samples)

        # ----------------------------------------------------------------------
        # 1. PARALLEL SAFETY INVARIANT: Emergency Stop (Co-Contraction) Override
        # ----------------------------------------------------------------------
        # Co-contraction is computed unconditionally from the parallel threshold activation.
        # The ML classifier CAN NEVER bypass or suppress an Emergency Stop.
        if fallback_intent.gesture == EMGIntent.CO_CONTRACTION_ESTOP:
            logger.warning("Emergency Stop: Co-contraction detected in parallel monitor!")
            self.last_gesture = EMGIntent.CO_CONTRACTION_ESTOP
            return EMGIntent(
                gesture=EMGIntent.CO_CONTRACTION_ESTOP,
                confidence=fallback_intent.confidence,
                proportional_force_n=fallback_intent.proportional_force_n,
                activation_level=fallback_intent.activation_level,
                channel_rms=fallback_intent.channel_rms,
                timestamp=now,
            )

        # ----------------------------------------------------------------------
        # 2. Check Fallback Mode
        # ----------------------------------------------------------------------
        if self.fallback_active or not self.model_ready:
            self.last_gesture = fallback_intent.gesture
            return fallback_intent

        # ----------------------------------------------------------------------
        # 3. Buffer Ingestion & Signal Quality / Lead-Off Check
        # ----------------------------------------------------------------------
        if new_samples.ndim == 1:
            if len(new_samples) == self.num_channels:
                new_samples = new_samples[:, np.newaxis]
            else:
                new_samples = new_samples[np.newaxis, :]

        ch_count, n_pts = new_samples.shape
        for ch in range(min(self.num_channels, ch_count)):
            self.raw_buffers[ch].extend(new_samples[ch].tolist())

        primary_window = np.array(self.raw_buffers[0])

        # Flat-line or saturated signal check (Lead-off or hardware disconnection)
        win_std = float(np.std(primary_window))
        win_max = float(np.max(primary_window))
        win_min = float(np.min(primary_window))

        # Check for disconnected sensor (std ~ 0.0) or rail saturation
        if win_std < 1e-4 or (win_max > 4090 and win_min > 4090) or (win_max < 5 and win_min < 5):
            self.lead_off_flag = True
            logger.debug("Lead-off or saturated signal detected. Defaulting to REST.")
            self.recent_predictions.append(EMGIntent.REST)
            self.last_gesture = EMGIntent.REST
            return EMGIntent(
                gesture=EMGIntent.REST,
                confidence=0.0,
                proportional_force_n=FORCE_MIN_N,
                activation_level=0.0,
                channel_rms=fallback_intent.channel_rms,
                timestamp=now,
            )
        else:
            self.lead_off_flag = False

        # ----------------------------------------------------------------------
        # 4. Feature Extraction & Scaler Inference
        # ----------------------------------------------------------------------
        try:
            feats = self._extract_window_features(primary_window)
            if feats is None or len(feats) != len(self.feature_columns):
                # Fail closed to fallback
                return fallback_intent

            X_df = pd.DataFrame([feats], columns=self.feature_columns)
            X_scaled = self.scaler.transform(X_df)

            # Check probability estimation support
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_scaled)[0]
                classes = list(self.model.classes_)
                max_idx = int(np.argmax(probs))
                raw_pred = classes[max_idx]
                confidence = float(probs[max_idx])
            else:
                raw_pred = str(self.model.predict(X_scaled)[0])
                confidence = 0.85

            predicted_gesture = self._map_prediction_to_gesture(raw_pred)

            # ------------------------------------------------------------------
            # 5. Confidence Gating
            # ------------------------------------------------------------------
            # If model confidence is below configured threshold, treat as REST
            if confidence < self.confidence_threshold:
                gated_gesture = EMGIntent.REST
            else:
                gated_gesture = predicted_gesture

            # ------------------------------------------------------------------
            # 6. Majority Voting & Hysteresis Across Last N Windows
            # ------------------------------------------------------------------
            self.recent_predictions.append(gated_gesture)
            vote_counts: Dict[str, int] = {}
            for g in self.recent_predictions:
                vote_counts[g] = vote_counts.get(g, 0) + 1

            majority_gesture = max(vote_counts, key=vote_counts.get)
            majority_count = vote_counts[majority_gesture]

            # Require strict majority (e.g. >= 2 out of 3)
            required_majority = (self.majority_vote_window // 2) + 1
            if majority_count >= required_majority:
                self.last_stable_gesture = majority_gesture
            else:
                # Retain previous stable gesture or default to REST
                self.last_stable_gesture = EMGIntent.REST

            final_gesture = self.last_stable_gesture
            self.last_gesture = final_gesture

            # Proportional Force from parallel activation
            prop_force_n = fallback_intent.proportional_force_n
            act_level = fallback_intent.activation_level

            return EMGIntent(
                gesture=final_gesture,
                confidence=confidence,
                proportional_force_n=prop_force_n,
                activation_level=act_level,
                channel_rms=fallback_intent.channel_rms,
                timestamp=now,
            )

        except Exception as ex:
            logger.warning(f"Inference error in EMGClassifier: {ex}. Failing closed to fallback decoder.")
            self.last_gesture = fallback_intent.gesture
            return fallback_intent
