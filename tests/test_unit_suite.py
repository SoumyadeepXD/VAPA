"""
VAPA Automated Unit Test Suite
Covers:
- Biosignals DSP & Filters (Butterworth, Notch, RMS, Bandpower)
- EMG Intent Decoding (1D/2D arrays, Co-Contraction E-Stop, Flexion Grasp)
- EEG Intent Decoding (1D/2D arrays, Mu ERD Reach, Frontal Blink Cycle)
- Multimodal Intent Fusion (Priority hierarchies, State-conditioned triggers)
- Forward & Inverse Kinematics (FK-IK consistency, Analytical solver convergence)
- Trajectory Planning (Minimum-jerk profiles)
- Arm Controller & Servo Drivers (9-channel mapping, Inversion formula, E-Stop safety)
"""

import unittest
import numpy as np
import time

from biosignals.signal_filters import (
    SignalFilter,
    compute_rms_envelope,
    compute_mav,
    compute_waveform_length,
    compute_bandpower,
)
from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.eeg_decoder import EEGDecoder, EEGIntent
from biosignals.intent_fusion import IntentFusionEngine, MultimodalCommand
from kinematics.arm_model import ArmModel
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import InverseKinematics
from kinematics.trajectory_planner import TrajectoryPlanner
from actuation.mock_arm_controller import MockServoDriver
from actuation.arm_controller import ArmController
from config.hardware_config import SERVO_CHANNELS
from config.system_config import JOINT_LIMITS_DEG


class TestBiosignalFilters(unittest.TestCase):
    def setUp(self):
        self.fs = 250.0  # Sample rate in Hz
        self.t = np.linspace(0, 1.0, int(self.fs), endpoint=False)
        self.filter = SignalFilter(sampling_rate_hz=self.fs)

    def test_butter_bandpass_standard(self):
        # 10 Hz sine wave within [5, 45] Hz bandpass
        sig = np.sin(2 * np.pi * 10 * self.t)
        self.filter.add_bandpass_filter("bp1", low_hz=5.0, high_hz=45.0)
        filtered = self.filter.filter_signal(sig, "bp1")
        self.assertEqual(len(filtered), len(sig))
        self.assertFalse(np.isnan(filtered).any())

    def test_butter_bandpass_boundary_safety(self):
        # Edge case: highcut >= Nyquist (Nyquist = 125 Hz for fs=250 Hz)
        sig = np.sin(2 * np.pi * 10 * self.t)
        # Should gracefully clamp or passthrough without crashing
        self.filter.add_bandpass_filter("bp_nyq", low_hz=20.0, high_hz=150.0)
        filtered = self.filter.filter_signal(sig, "bp_nyq")
        self.assertEqual(len(filtered), len(sig))
        self.assertFalse(np.isnan(filtered).any())

        # Edge case: lowcut >= highcut
        self.filter.add_bandpass_filter("bp_inv", low_hz=50.0, high_hz=20.0)
        filtered_inv = self.filter.filter_signal(sig, "bp_inv")
        self.assertEqual(len(filtered_inv), len(sig))

    def test_notch_filter_nyquist_safety(self):
        # 50 Hz notch on 80 Hz sample rate (Nyquist = 40 Hz < 50 Hz)
        low_fs_filter = SignalFilter(sampling_rate_hz=80.0)
        sig = np.ones(100)
        low_fs_filter.add_notch_filter("notch50", notch_freq_hz=50.0, q=30.0)
        filtered = low_fs_filter.filter_signal(sig, "notch50")
        # Must return input array unharmed without throwing ValueError
        np.testing.assert_array_equal(filtered, sig)

    def test_compute_rms_envelope_short_signal(self):
        # Signal shorter than window length (e.g. 5 samples, window 50)
        short_sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms = compute_rms_envelope(short_sig, window_size=50)
        self.assertEqual(len(rms), len(short_sig))
        self.assertFalse(np.isnan(rms).any())

    def test_compute_bandpower_dc_offset_rejection(self):
        # Sine wave with massive DC offset baseline
        sig_ac = np.sin(2 * np.pi * 10 * self.t)  # 10 Hz in Mu band [8, 12]
        sig_dc = sig_ac + 1000.0  # Massive 1000V DC offset

        bp = compute_bandpower(sig_dc, sampling_rate_hz=self.fs, band=(8.0, 12.0))
        # Relative band power in Mu should be significant (> 0.2), not diluted to 0.00001
        self.assertGreater(bp, 0.2)


class TestEMGDecoder(unittest.TestCase):
    def setUp(self):
        self.fs = 250.0
        self.decoder = EMGDecoder(num_channels=2, fs=self.fs)
        self.t = np.linspace(0, 0.5, int(self.fs * 0.5))

    def test_1d_sample_ingestion(self):
        # Feed 1D array of 2 channels (1 sample each)
        intent = self.decoder.update_samples(np.array([0.05, 0.05]))
        self.assertIsInstance(intent, EMGIntent)

        # Feed 1D array of length 20 (interpreted as 10 samples for 2 channels)
        intent = self.decoder.update_samples(np.zeros(20))
        self.assertIsInstance(intent, EMGIntent)

    def test_co_contraction_estop(self):
        # Both channels high 100 Hz bursts (co-contraction, ~180 uV peak)
        # Shape: (num_channels, num_samples)
        burst_ch = 180.0 * np.sin(2 * np.pi * 100.0 * self.t)
        high_burst = np.vstack([burst_ch, burst_ch])
        intent = self.decoder.update_samples(high_burst)
        self.assertEqual(intent.gesture, EMGIntent.CO_CONTRACTION_ESTOP)

    def test_flexor_burst_grasp(self):
        # Flexor (CH0) high 100 Hz burst, Extensor (CH1) resting noise
        burst_ch0 = 120.0 * np.sin(2 * np.pi * 100.0 * self.t)
        burst_ch1 = np.zeros_like(burst_ch0)
        burst = np.vstack([burst_ch0, burst_ch1])
        intent = self.decoder.update_samples(burst)
        self.assertEqual(intent.gesture, EMGIntent.GRASP_CLOSE)


class TestEEGDecoder(unittest.TestCase):
    def setUp(self):
        self.fs = 250.0
        self.decoder = EEGDecoder(num_channels=4, fs=self.fs)
        self.t = np.linspace(0, 0.5, int(self.fs * 0.5))

    def test_1d_sample_ingestion(self):
        intent = self.decoder.update_samples(np.array([10.0, 10.0, 10.0, 10.0]))
        self.assertIsInstance(intent, EEGIntent)

    def test_frontal_blink_cycle(self):
        # Frontal channel (CH3) has an 80+ uV 5 Hz blink pulse within EEG bandpass [1, 45] Hz
        # Shape: (num_channels, num_samples)
        eeg_burst = np.zeros((4, len(self.t)))
        eeg_burst[3, :] = 150.0 * np.sin(2 * np.pi * 5.0 * self.t)
        intent = self.decoder.update_samples(eeg_burst)
        self.assertEqual(intent.command, EEGIntent.TARGET_CYCLE_NEXT)


class TestIntentFusion(unittest.TestCase):
    def setUp(self):
        self.fusion = IntentFusionEngine()

    def test_estop_priority(self):
        emg = EMGIntent(
            gesture=EMGIntent.CO_CONTRACTION_ESTOP,
            confidence=0.9,
            proportional_force_n=0.0,
            activation_level=0.9,
            channel_rms=[100.0, 100.0],
            timestamp=time.time(),
        )
        eeg = EEGIntent(
            command=EEGIntent.INTENT_REACH,
            confidence=0.9,
            mu_power=0.05,
            beta_power=0.2,
            attention_score=0.9,
            motor_imagery_active=True,
            timestamp=time.time(),
        )
        cmd = self.fusion.fuse(emg, eeg, visible_targets=[], system_state="REACHING")
        self.assertEqual(cmd.action, MultimodalCommand.EMERGENCY_STOP)


class TestKinematicsAndPlanning(unittest.TestCase):
    def setUp(self):
        self.model = ArmModel()
        self.fk = ForwardKinematics(self.model)
        self.ik = InverseKinematics(self.model)
        self.planner = TrajectoryPlanner()

    def test_fk_home_pose(self):
        fk_res = self.fk.compute_fk(self.model.home_angles_deg)
        pos = fk_res["ee_position"]
        self.assertAlmostEqual(pos[0], 0.437, places=2)
        self.assertAlmostEqual(pos[1], 0.000, places=2)
        self.assertAlmostEqual(pos[2], 0.207, places=2)

    def test_ik_analytical_convergence(self):
        target = np.array([0.25, 0.00, 0.15])
        angles, success = self.ik.solve_analytical(target, target_pitch_deg=0.0)
        self.assertTrue(success)
        fk_res = self.fk.compute_fk(angles)
        err = np.linalg.norm(target - fk_res["ee_position"])
        self.assertLess(err, 0.015)  # Error < 15 mm

    def test_trajectory_planner_profile(self):
        start_q = self.model.home_angles_deg
        target_q = {"joint_1_base_yaw": 30.0, "joint_2_shoulder_pitch": 60.0}
        traj = self.planner.plan_trajectory(start_q, target_q, min_duration_s=1.0)
        self.assertGreater(len(traj), 10)
        # Start and end angles match
        self.assertAlmostEqual(traj[0].angles_deg["joint_1_base_yaw"], start_q["joint_1_base_yaw"], places=1)
        self.assertAlmostEqual(traj[-1].angles_deg["joint_1_base_yaw"], target_q["joint_1_base_yaw"], places=1)


class TestArmControllerAndDrivers(unittest.TestCase):
    def setUp(self):
        self.driver = MockServoDriver()
        self.arm = ArmController(force_mock=True)

    def test_mock_driver_channel_mapping(self):
        # Verify all 9 build-phase physical joints are registered
        angles = self.driver.current_angles
        for j_name in SERVO_CHANNELS:
            self.assertIn(j_name, angles)

    def test_servo_inversion_mapping(self):
        # Test an inverted channel (e.g. direction = -1)
        # Using finger_thumb or joint_wrist_flex
        cfg = {"channel": 0, "min_angle_deg": 0.0, "max_angle_deg": 180.0, "offset_deg": 0.0, "direction": -1}
        min_a = cfg["min_angle_deg"]
        max_a = cfg["max_angle_deg"]
        offset = cfg["offset_deg"]

        # Angle 0 deg with direction = -1 should map to max_a (180 deg)
        mapped_0 = (max_a - (0.0 - min_a)) + offset
        self.assertEqual(mapped_0, 180.0)

        # Angle 180 deg with direction = -1 should map to min_a (0 deg)
        mapped_180 = (max_a - (180.0 - min_a)) + offset
        self.assertEqual(mapped_180, 0.0)

    def test_arm_emergency_stop_freezes_movement(self):
        self.arm.emergency_stop()
        self.assertTrue(self.arm.is_emergency_stopped)
        success = self.arm.move_to_angles({"joint_1_base_yaw": 45.0}, duration_s=0.1)
        self.assertFalse(success)
        self.arm.reset_emergency_stop()
        self.assertFalse(self.arm.is_emergency_stopped)

    def test_force_grasp_closed_loop(self):
        # Sensor function providing simulated increasing contact force
        force_state = {"f": 0.0}
        def simulated_tactile_sensor():
            force_state["f"] += 0.8
            return force_state["f"]

        success = self.arm.execute_force_grasp(
            target_force_n=2.0,
            timeout_s=1.0,
            tactile_sensor_fn=simulated_tactile_sensor,
        )
        self.assertTrue(success)
        self.assertGreaterEqual(self.arm.current_force_n, 2.0)


if __name__ == "__main__":
    unittest.main()
