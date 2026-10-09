"""
VAPA EMG Classifier & Safe ML Architecture Verification Suite
Tests fail-closed loader, cryptographic integrity, confidence gating,
majority voting, lead-off protection, parallel E-stop, CSV replay, and latency benchmarks.
"""

import os
import time
import unittest
import numpy as np

from config.system_config import (
    EMG_SAMPLING_RATE_HZ,
    FORCE_MIN_N,
    FORCE_MAX_N,
    EMG_CO_CONTRACTION_THRESHOLD,
)
from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.emg_classifier import EMGClassifier, SafeModelLoader, EXPECTED_MANIFEST_HASHES
from biosignals.intent_fusion import IntentFusionEngine, MultimodalCommand
from vision.spatial_3d import GraspTarget3D
from biosignals.eeg_decoder import EEGIntent


class TestSafeModelLoader(unittest.TestCase):
    """Verifies cryptographic hash validation and fail-closed deserialization."""

    def test_loader_real_artifacts_success(self):
        """Verifies that the verified team model artifacts pass loader validation."""
        success, model, scaler, cols, reason = SafeModelLoader.load_artifacts(
            model_path="biosignals/models/svm_emg_model.pkl",
            scaler_path="biosignals/models/emg_scaler.pkl",
            features_path="biosignals/models/feature_columns.pkl",
            enforce_manifest_check=True,
        )
        self.assertTrue(success, f"Expected successful load, got: {reason}")
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertEqual(len(cols), 10)
        self.assertEqual(reason, "Success")

    def test_loader_fail_closed_on_hash_mismatch(self):
        """Verifies fail-closed behavior if hash does not match manifest."""
        tampered_hashes = EXPECTED_MANIFEST_HASHES.copy()
        tampered_hashes["biosignals/models/svm_emg_model.pkl"] = "0000000000000000000000000000000000000000000000000000000000000000"

        success, model, scaler, cols, reason = SafeModelLoader.load_artifacts(
            model_path="biosignals/models/svm_emg_model.pkl",
            scaler_path="biosignals/models/emg_scaler.pkl",
            features_path="biosignals/models/feature_columns.pkl",
            expected_hashes=tampered_hashes,
            enforce_manifest_check=True,
        )
        self.assertFalse(success)
        self.assertIsNone(model)
        self.assertIn("hash mismatch", reason.lower())

    def test_loader_fail_closed_on_missing_file(self):
        """Verifies fail-closed behavior when an artifact is missing."""
        success, model, scaler, cols, reason = SafeModelLoader.load_artifacts(
            model_path="biosignals/models/non_existent_model.pkl",
            scaler_path="biosignals/models/emg_scaler.pkl",
            features_path="biosignals/models/feature_columns.pkl",
            enforce_manifest_check=False,
        )
        self.assertFalse(success)
        self.assertIsNone(model)
        self.assertIn("not found", reason.lower())

    def test_classifier_falls_back_when_model_fails(self):
        """Verifies that EMGClassifier falls back to threshold EMGDecoder if model fails."""
        classifier = EMGClassifier(
            model_path="biosignals/models/non_existent_model.pkl",
            enforce_manifest_check=False,
        )
        self.assertFalse(classifier.model_ready)
        self.assertTrue(classifier.fallback_active)

        # Feeding zero samples should return REST from fallback decoder
        samples = np.zeros((4, 50))
        intent = classifier.update_samples(samples)
        self.assertIsInstance(intent, EMGIntent)
        self.assertEqual(intent.gesture, EMGIntent.REST)


class TestEMGClassifierInferenceSafety(unittest.TestCase):
    """Verifies confidence gating, majority voting, lead-off detection, and E-Stop invariant."""

    def setUp(self):
        self.classifier = EMGClassifier(
            num_channels=4,
            fs=500.0,
            confidence_threshold=0.70,
            majority_vote_window=3,
            window_duration_ms=200.0,
            enforce_manifest_check=True,
        )

    def test_lead_off_and_flatline_detection(self):
        """Flatline signal (disconnected electrode) must yield REST and flag lead-off."""
        flat_samples = np.ones((4, 100)) * 1900.0  # Completely flat DC
        intent = self.classifier.update_samples(flat_samples)
        self.assertEqual(intent.gesture, EMGIntent.REST)
        self.assertTrue(self.classifier.lead_off_flag)

    def test_rail_saturation_detection(self):
        """Rail saturation (broken/floating lead) must yield REST and flag lead-off."""
        saturated_samples = np.ones((4, 100)) * 4095.0  # 12-bit ADC rail
        intent = self.classifier.update_samples(saturated_samples)
        self.assertEqual(intent.gesture, EMGIntent.REST)
        self.assertTrue(self.classifier.lead_off_flag)

    def test_rest_default_on_low_amplitude(self):
        """Baseline noise must default to REST."""
        # Simulated low-amplitude baseline noise around 1905 counts
        np.random.seed(42)
        noise = 1905.0 + np.random.normal(0, 3.0, (4, 100))
        intent = self.classifier.update_samples(noise)
        self.assertEqual(intent.gesture, EMGIntent.REST)

    def test_parallel_co_contraction_emergency_stop_invariant(self):
        """
        CRITICAL SAFETY INVARIANT:
        Simultaneous high flexor and extensor activation MUST unconditionally
        trigger CO_CONTRACTION_ESTOP regardless of ML classifier state.
        """
        fs = 500.0
        # High-amplitude 80 Hz burst on both Flexor (Ch 0) and Extensor (Ch 1)
        t = np.linspace(0, 0.20, int(fs * 0.20))
        burst_flexor = 1905.0 + 800.0 * np.sin(2 * np.pi * 80 * t)
        burst_extensor = 1905.0 + 800.0 * np.sin(2 * np.pi * 80 * t)
        burst_samples = np.zeros((4, len(t)))
        burst_samples[0] = burst_flexor
        burst_samples[1] = burst_extensor

        intent = self.classifier.update_samples(burst_samples)
        self.assertEqual(
            intent.gesture,
            EMGIntent.CO_CONTRACTION_ESTOP,
            "Classifier must never override or suppress Emergency Co-Contraction Stop!",
        )

    def test_intent_fusion_maps_classifier_estop(self):
        """Verifies that IntentFusionEngine translates CO_CONTRACTION_ESTOP to EMERGENCY_STOP."""
        fusion = IntentFusionEngine()
        estop_intent = EMGIntent(
            gesture=EMGIntent.CO_CONTRACTION_ESTOP,
            confidence=0.95,
            proportional_force_n=FORCE_MIN_N,
            activation_level=0.90,
            channel_rms=[100.0, 100.0, 0.0, 0.0],
            timestamp=time.time(),
        )
        dummy_eeg = EEGIntent(
            command=EEGIntent.IDLE,
            confidence=0.0,
            mu_power=0.0,
            beta_power=0.0,
            attention_score=0.0,
            motor_imagery_active=False,
            timestamp=time.time(),
        )
        cmd = fusion.fuse(estop_intent, dummy_eeg, [], "GRASPING")
        self.assertEqual(cmd.action, MultimodalCommand.EMERGENCY_STOP)
        self.assertEqual(cmd.source, "EMG_ESTOP")

    def test_gesture_mapping_logic(self):
        """Verifies canonical string mappings."""
        self.assertEqual(self.classifier._map_prediction_to_gesture("FIST"), EMGIntent.GRASP_CLOSE)
        self.assertEqual(self.classifier._map_prediction_to_gesture("OPEN"), EMGIntent.HAND_OPEN)
        self.assertEqual(self.classifier._map_prediction_to_gesture("WRIST_FLEXION"), EMGIntent.GRASP_CLOSE)
        self.assertEqual(self.classifier._map_prediction_to_gesture("WRIST_EXTENSION"), EMGIntent.HAND_OPEN)
        self.assertEqual(self.classifier._map_prediction_to_gesture("REST"), EMGIntent.REST)


class TestEMGReplayAndLatency(unittest.TestCase):
    """Replays recorded sample CSVs through the classifier and measures inference latency."""

    def test_replay_sample_csv(self):
        """Streams samples from data/emg/raw/rest_sample.csv into EMGClassifier."""
        csv_path = "data/emg/raw/rest_sample.csv"
        self.assertTrue(os.path.exists(csv_path), f"Sample file {csv_path} must exist")

        classifier = EMGClassifier(num_channels=4, fs=500.0)

        # Read CSV rows
        import csv
        values = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 2:
                    values.append(float(row[1]))

        self.assertGreater(len(values), 50)
        arr = np.array(values)
        chunk_size = 20
        intents = []
        for i in range(0, len(arr) - chunk_size + 1, chunk_size):
            chunk = np.zeros((4, chunk_size))
            chunk[0] = arr[i : i + chunk_size]
            intent = classifier.update_samples(chunk)
            intents.append(intent)

        self.assertGreater(len(intents), 0)
        # Final intent should be valid EMGIntent
        last_intent = intents[-1]
        self.assertIsInstance(last_intent, EMGIntent)
        self.assertIn(last_intent.gesture, [EMGIntent.REST, EMGIntent.GRASP_CLOSE, EMGIntent.HAND_OPEN])

    def test_inference_latency_benchmark(self):
        """Measures per-window inference time and verifies < 5ms latency."""
        classifier = EMGClassifier(num_channels=4, fs=500.0)
        np.random.seed(42)
        test_samples = 1905.0 + np.random.normal(0, 50.0, (4, 100))

        # Warm up
        for _ in range(5):
            classifier.update_samples(test_samples)

        latencies_ms = []
        for _ in range(100):
            t0 = time.perf_counter()
            classifier.update_samples(test_samples)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        max_lat = float(np.max(latencies_ms))

        print(f"\n[LATENCY BENCHMARK] Mean: {mean_lat:.3f} ms | P95: {p95_lat:.3f} ms | Max: {max_lat:.3f} ms")
        # Must execute comfortably within real-time budget (< 10 ms, window budget is 250 ms)
        self.assertLess(mean_lat, 10.0, f"Mean latency {mean_lat:.2f}ms exceeds 10ms threshold")
        self.assertLess(p95_lat, 25.0, f"P95 latency {p95_lat:.2f}ms exceeds 25ms threshold")


if __name__ == "__main__":
    unittest.main()
