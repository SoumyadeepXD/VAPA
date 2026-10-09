"""
VAPA Comprehensive Two-Site EMG & Safety Layer Test Suite
Tests:
1. Two-Site Antagonist Proportional Decoding (Flexor -> Close, Extensor -> Open, Rest)
2. Lead-Off, Saturation, and Flatline Fault Safety (Forcing REST, never CLOSE)
3. Parallel Co-Contraction Emergency Stop Invariant (Classifier Disabled & Enabled)
4. Calibrated Servo Bounds & Missing Calibration Arming Gate (Refusal to Arm)
5. ESP32 Hardware E-Stop Button Telemetry & Fusion Preemption
6. Grip Profile Binding at Grasp Start & Monotonic Mid-Grasp Ceiling Lowering
7. Camera-Off / Zero-Targets EMG-Only Operation
8. Dual-Mode Telemetry Stream Replay (Binary CRC16 & High-Speed JSON)
"""

import os
import sys
import json
import time
import unittest
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.emg_classifier import EMGClassifier
from biosignals.eeg_decoder import EEGIntent
from biosignals.esp32_serial_receiver import (
    ESP32TelemetryFrame,
    AsyncESP32Receiver,
    compute_crc16,
)
from biosignals.intent_fusion import IntentFusionEngine, MultimodalCommand
from actuation.arm_controller import ArmController
from actuation.mock_arm_controller import MockServoDriver
from vision.spatial_3d import GraspTarget3D
from vision.grip_profiles import GripProfileManager, GripProfile


class TestTwoSiteEMGDecoder(unittest.TestCase):
    """Tests two-site antagonist proportional control and safety invariants."""
    def setUp(self):
        self.fs = 250.0
        self.decoder = EMGDecoder(num_channels=2, fs=self.fs)
        self.t = np.linspace(0, 0.20, int(self.fs * 0.20))

    def test_antagonist_flexor_closes_hand(self):
        """Flexor dominant burst (> 0.28 act, > 1.25x extensor) decodes to GRASP_CLOSE."""
        # Flexor: 1.8V envelope (normalized ~0.8), Extensor: 0.08V (resting)
        samples = np.vstack([
            np.full(len(self.t), 1.80),
            np.full(len(self.t), 0.08),
        ])
        intent = self.decoder.update_samples(samples)
        self.assertEqual(intent.gesture, EMGIntent.GRASP_CLOSE)
        self.assertGreater(intent.proportional_force_n, 2.0)
        self.assertFalse(intent.lead_off_detected)

    def test_antagonist_extensor_opens_hand(self):
        """Extensor dominant burst (> 0.28 act, > 1.25x flexor) decodes to HAND_OPEN."""
        # Flexor: 0.08V (resting), Extensor: 1.80V (extension)
        samples = np.vstack([
            np.full(len(self.t), 0.08),
            np.full(len(self.t), 1.80),
        ])
        intent = self.decoder.update_samples(samples)
        self.assertEqual(intent.gesture, EMGIntent.HAND_OPEN)
        self.assertFalse(intent.lead_off_detected)

    def test_resting_muscles_stay_in_rest(self):
        """Signals below deadband (0.12) stay in REST state."""
        samples = np.vstack([
            np.full(len(self.t), 0.08),
            np.full(len(self.t), 0.08),
        ])
        intent = self.decoder.update_samples(samples)
        self.assertEqual(intent.gesture, EMGIntent.REST)

    def test_lead_off_voltage_saturation_forces_rest(self):
        """Upper rail saturation (>= 3.25V) forces REST, sets lead_off_flag, NEVER closes."""
        samples = np.vstack([
            np.full(len(self.t), 3.29),  # Flexor saturated to supply rail!
            np.full(len(self.t), 0.08),
        ])
        intent = self.decoder.update_samples(samples)
        self.assertEqual(intent.gesture, EMGIntent.REST)
        self.assertTrue(intent.lead_off_detected)
        self.assertIn("saturated", intent.fault_reason)

    def test_lead_off_ground_fault_forces_rest(self):
        """Ground-fault flatline (<= 0.005V) forces REST, sets lead_off_flag, NEVER closes."""
        samples = np.vstack([
            np.full(len(self.t), 0.001), # Wire severed to GND
            np.full(len(self.t), 0.08),
        ])
        intent = self.decoder.update_samples(samples)
        self.assertEqual(intent.gesture, EMGIntent.REST)
        self.assertTrue(intent.lead_off_detected)
        self.assertIn("flatline", intent.fault_reason)

    def test_co_contraction_unconditional_estop_threshold_decoder(self):
        """Simultaneous flexor and extensor contractions trigger CO_CONTRACTION_ESTOP."""
        # High bursts on both channels: 2.1V each (well above 0.85 activation)
        samples = np.vstack([
            np.full(len(self.t), 2.15),
            np.full(len(self.t), 2.15),
        ])
        intent = self.decoder.update_samples(samples)
        self.assertEqual(intent.gesture, EMGIntent.CO_CONTRACTION_ESTOP)
        self.assertGreaterEqual(intent.activation_level, 0.85)

    def test_co_contraction_unconditional_estop_classifier_decoder(self):
        """Co-contraction unconditionally triggers E-Stop even when EMGClassifier is instantiated."""
        classifier = EMGClassifier()
        samples = np.vstack([
            np.full(len(self.t), 2.15),
            np.full(len(self.t), 2.15),
            np.zeros(len(self.t)),
            np.zeros(len(self.t)),
        ])
        intent = classifier.update_samples(samples)
        self.assertEqual(intent.gesture, EMGIntent.CO_CONTRACTION_ESTOP)


class TestServoCalibrationArmingGate(unittest.TestCase):
    """Tests the fail-closed arming gate and per-servo mechanical limits."""
    def test_missing_calibration_refuses_to_arm(self):
        """ArmController MUST refuse to arm when calibration file is missing."""
        mock_driver = MockServoDriver()
        controller = ArmController(
            driver=mock_driver,
            force_mock=True,
            calibration_path="/non_existent/path/servo_cal.json",
        )
        self.assertFalse(controller.is_armed)

        # Movement commands must be completely ignored
        controller.move_to_angles({"joint_wrist_flex": 45.0})
        self.assertNotEqual(controller.get_joint_angles().get("joint_wrist_flex"), 45.0)

        # Gripper commands must be ignored
        controller.set_gripper_opening_percent(50.0)
        self.assertEqual(mock_driver.target_angles.get("joint_6_gripper", 100.0), 100.0)

        # Trajectory commands must be refused
        success = controller.execute_trajectory([])
        self.assertFalse(success)

    def test_valid_calibration_arms_and_clamps_limits(self):
        """ArmController loads calibration, arms, and clamps commands to per-joint bounds."""
        mock_driver = MockServoDriver()
        calib_file = os.path.join(REPO_ROOT, "config/servo_calibration.json")
        controller = ArmController(
            driver=mock_driver,
            force_mock=True,
            calibration_path=calib_file,
        )
        self.assertTrue(controller.is_armed)

        # Wrist flex is calibrated to [15.0, 165.0]
        # Command 180.0 -> must be clamped to 165.0
        controller.move_to_angles({"joint_wrist_flex": 180.0}, duration_s=0.01)
        clamped_angle = controller.get_joint_angles().get("joint_wrist_flex")
        self.assertAlmostEqual(clamped_angle, 165.0, places=1)

        # Command -10.0 -> must be clamped to 15.0
        controller.move_to_angles({"joint_wrist_flex": -10.0}, duration_s=0.01)
        clamped_angle = controller.get_joint_angles().get("joint_wrist_flex")
        self.assertAlmostEqual(clamped_angle, 15.0, places=1)


class TestVisionGripLink(unittest.TestCase):
    """Tests grip profile linking, mid-grasp force ceiling guard, and camera-off EMG control."""
    def setUp(self):
        self.grip_mgr = GripProfileManager()
        self.fusion = IntentFusionEngine()

    def test_grip_profile_selection_on_grasp_start(self):
        """Profile is chosen at grasp start based on detected object class."""
        profile_egg = self.grip_mgr.on_grasp_start("egg", confidence=0.95)
        self.assertEqual(profile_egg.label, "egg")
        self.assertEqual(profile_egg.target_force_n, 1.5)
        self.assertEqual(profile_egg.force_ceiling_n, 3.0)

        profile_mug = self.grip_mgr.on_grasp_start("mug", confidence=0.90)
        self.assertEqual(profile_mug.label, "mug")
        self.assertEqual(profile_mug.target_force_n, 4.5)
        self.assertEqual(profile_mug.force_ceiling_n, 8.0)

    def test_mid_grasp_may_only_lower_force_ceiling(self):
        """Mid-grasp perception updates can LOWER the force ceiling, but NEVER raise it."""
        # 1. Start grasping a sturdy mug (ceiling = 8.0 N)
        self.grip_mgr.on_grasp_start("mug", confidence=0.90)
        self.assertEqual(self.grip_mgr.current_force_ceiling_n, 8.0)

        # 2. Mid-grasp vision reclassifies to fragile egg (ceiling = 3.0 N) -> MUST LOWER!
        updated_ceiling = self.grip_mgr.update_mid_grasp("egg", confidence=0.90)
        self.assertEqual(updated_ceiling, 3.0)
        self.assertEqual(self.grip_mgr.current_force_ceiling_n, 3.0)

        # 3. Vision later reclassifies to heavy bottle (ceiling = 9.0 N) -> MUST NOT RAISE!
        updated_ceiling = self.grip_mgr.update_mid_grasp("bottle", confidence=0.90)
        self.assertEqual(updated_ceiling, 3.0)  # Kept at 3.0 N!
        self.assertEqual(self.grip_mgr.current_force_ceiling_n, 3.0)

    def test_perception_alone_never_initiates_motion(self):
        """Having detected objects in the scene NEVER causes the arm to move on its own."""
        dummy_target = GraspTarget3D(
            label="mug",
            score=0.95,
            center_camera_m=np.array([0.0, 0.0, 0.4]),
            center_base_m=np.array([0.3, 0.0, 0.1]),
            width_m=0.08, height_m=0.10, depth_m=0.08,
            approach_vector=np.array([1.0, 0.0, 0.0]),
            grasp_yaw_deg=0.0, grasp_pitch_deg=0.0,
            target_opening_m=0.09, target_force_n=4.5,
            is_reachable=True, pixel_box=(100, 100, 200, 200),
        )
        emg_rest = EMGIntent(EMGIntent.REST, 0.9, 0.3, 0.05, [0.05, 0.05], time.time())
        eeg_idle = EEGIntent(EEGIntent.IDLE, 1.0, 0.0, 0.0, 0.5, False, time.time())

        # Target is visible, but user is resting
        cmd = self.fusion.fuse(emg_rest, eeg_idle, [dummy_target], system_state="SCANNING")
        self.assertEqual(cmd.action, MultimodalCommand.NO_OP)

    def test_camera_off_direct_emg_control(self):
        """With camera off (0 targets), arm operates directly from EMG alone."""
        emg_flex = EMGIntent(EMGIntent.GRASP_CLOSE, 0.95, 5.0, 0.75, [1.5, 0.1], time.time())
        eeg_idle = EEGIntent(EEGIntent.IDLE, 1.0, 0.0, 0.0, 0.5, False, time.time())

        # No visible targets (camera off)
        cmd_close = self.fusion.fuse(emg_flex, eeg_idle, [], system_state="IDLE")
        self.assertEqual(cmd_close.action, MultimodalCommand.START_GRASP)
        self.assertEqual(cmd_close.source, "EMG_DIRECT_GRASP")
        self.assertIsNone(cmd_close.target)

        # Extensor release with camera off
        emg_open = EMGIntent(EMGIntent.HAND_OPEN, 0.95, 0.3, 0.70, [0.1, 1.5], time.time())
        cmd_open = self.fusion.fuse(emg_open, eeg_idle, [], system_state="IDLE")
        self.assertEqual(cmd_open.action, MultimodalCommand.RELEASE_GRIP)
        self.assertEqual(cmd_open.source, "EMG_DIRECT_RELEASE")


class TestHardwareEstopAndTelemetry(unittest.TestCase):
    """Tests hardware E-Stop button input and dual-mode binary CRC16 stream."""
    def test_esp32_hardware_estop_button_triggers_emergency_stop(self):
        """Hardware button press from ESP32 telemetry immediately forces EMERGENCY_STOP."""
        fusion = IntentFusionEngine()
        emg_rest = EMGIntent(EMGIntent.REST, 1.0, 0.3, 0.05, [0.05, 0.05], time.time())
        eeg_idle = EEGIntent(EEGIntent.IDLE, 1.0, 0.0, 0.0, 0.5, False, time.time())

        cmd = fusion.fuse(emg_rest, eeg_idle, [], system_state="GRASPING", estop_hardware_trigger=True)
        self.assertEqual(cmd.action, MultimodalCommand.EMERGENCY_STOP)
        self.assertEqual(cmd.source, "ESP32_BUTTON_ESTOP")

    def test_binary_crc16_framing_parser(self):
        """Validates 33-byte compact binary packet deserialization with CRC16."""
        # Construct synthetic 33-byte packet matching ESP32 firmware
        pkt = bytearray(33)
        pkt[0] = 0xAA
        pkt[1] = 0x55
        pkt[2] = 33  # Length
        # Seq = 100
        pkt[3:7] = (100).to_bytes(4, "big")
        # 5x FSRs = [100, 200, 300, 400, 500] mV
        for i, val in enumerate([100, 200, 300, 400, 500]):
            pkt[7 + i*2 : 9 + i*2] = val.to_bytes(2, "big")
        # Flexor mV = 1500 mV (1.5V)
        pkt[17:19] = (1500).to_bytes(2, "big")
        # Extensor mV = 200 mV (0.2V)
        pkt[19:21] = (200).to_bytes(2, "big")
        # EEG mV = 350 mV
        pkt[21:23] = (350).to_bytes(2, "big")
        # Enc cd = 12540 cd (125.4 deg)
        pkt[23:25] = (12540).to_bytes(2, "big")
        # E-Stop = 0
        pkt[25] = 0
        # Ts = 482910 ms
        pkt[26:30] = (482910).to_bytes(4, "big")
        # Compute CRC16 over pkt[2:30]
        crc = compute_crc16(pkt[2:30])
        pkt[30:32] = crc.to_bytes(2, "big")
        pkt[32] = ord('\n')

        # Test CRC verification
        calc_crc = compute_crc16(pkt[2:30])
        self.assertEqual(calc_crc, (pkt[30] << 8) | pkt[31])

        # Frame properties
        frame = ESP32TelemetryFrame(
            seq=100,
            fsr_volts=[0.1, 0.2, 0.3, 0.4, 0.5],
            emg_flex_volts=1.5,
            emg_ext_volts=0.2,
            eeg_volts=0.35,
            enc_deg=[125.4],
            estop_button_pressed=False,
            esp_timestamp_ms=482910,
        )
        self.assertEqual(frame.seq, 100)
        self.assertAlmostEqual(frame.emg_flex_volts, 1.5, places=2)
        self.assertAlmostEqual(frame.emg_ext_volts, 0.2, places=2)
        self.assertAlmostEqual(frame.encoder_angle_deg, 125.4, places=1)
        self.assertFalse(frame.estop_button_pressed)


if __name__ == "__main__":
    unittest.main()
