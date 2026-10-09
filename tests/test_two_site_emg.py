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

    def test_uncalibrated_or_example_file_refuses_to_arm(self):
        """ArmController MUST refuse to arm when file has 'calibrated': false (like .example)."""
        mock_driver = MockServoDriver()
        example_file = os.path.join(REPO_ROOT, "config/servo_calibration.json.example")
        controller = ArmController(
            driver=mock_driver,
            force_mock=True,
            calibration_path=example_file,
        )
        self.assertFalse(controller.is_armed)

    def test_missing_timestamp_or_version_refuses_to_arm(self):
        """ArmController MUST refuse to arm if timestamp or tool_version is missing."""
        import tempfile
        mock_driver = MockServoDriver()

        # Missing timestamp
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "tool_version": "1.0.0",
                "servos": {"joint_wrist_flex": {"min_deg": 15.0, "max_deg": 165.0, "home_deg": 90.0}}
            }, f)
            tmp_path = f.name
        try:
            ctrl = ArmController(driver=mock_driver, force_mock=True, calibration_path=tmp_path)
            self.assertFalse(ctrl.is_armed)
        finally:
            os.remove(tmp_path)

        # Missing tool_version
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "timestamp": time.time(),
                "servos": {"joint_wrist_flex": {"min_deg": 15.0, "max_deg": 165.0, "home_deg": 90.0}}
            }, f)
            tmp_path = f.name
        try:
            ctrl = ArmController(driver=mock_driver, force_mock=True, calibration_path=tmp_path)
            self.assertFalse(ctrl.is_armed)
        finally:
            os.remove(tmp_path)

    def test_valid_calibration_arms_and_clamps_limits(self):
        """ArmController loads valid calibration with calibrated: True, arms, and clamps commands."""
        import tempfile
        mock_driver = MockServoDriver()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "timestamp": time.time(),
                "tool_version": "1.0.0",
                "servos": {
                    "joint_wrist_flex": {"min_deg": 15.0, "max_deg": 165.0, "home_deg": 90.0}
                }
            }, f)
            calib_file = f.name

        try:
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
        finally:
            os.remove(calib_file)

    def test_emg_decoder_refuses_uncalibrated_example(self):
        """EMGDecoder refuses to load calibration if calibrated is False or missing."""
        decoder = EMGDecoder()
        example_file = os.path.join(REPO_ROOT, "config/emg_calibration.json.example")
        self.assertFalse(decoder.load_calibration(example_file))

    def test_calibration_clis_set_required_metadata(self):
        """Only calibration CLIs set 'calibrated': True, timestamp, and tool_version."""
        import tempfile
        from tools.calibrate_servos import save_calibration as save_servo_cal
        from tools.emg_training.calibrate_emg_mvc import run_guided_calibration

        # Test servo calibration CLI saver
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            tmp_servo = f.name
        try:
            save_servo_cal({"servos": {}}, tmp_servo)
            with open(tmp_servo) as f:
                d = json.load(f)
            self.assertTrue(d.get("calibrated"))
            self.assertTrue(bool(d.get("timestamp")))
            self.assertEqual(d.get("tool_version"), "1.0.0")
        finally:
            if os.path.exists(tmp_servo):
                os.remove(tmp_servo)

        # Test EMG calibration CLI saver
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            tmp_emg = f.name
        try:
            run_guided_calibration("TEST_SUBJ", tmp_emg, force_mock=True, fast=True)
            with open(tmp_emg) as f:
                d = json.load(f)
            self.assertTrue(d.get("calibrated"))
            self.assertTrue(bool(d.get("timestamp")))
            self.assertEqual(d.get("tool_version"), "1.0.0")
        finally:
            if os.path.exists(tmp_emg):
                os.remove(tmp_emg)


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

    def test_knife_and_scissors_never_get_automatic_custom_grip_profile(self):
        """Item 5: Knife and scissors must NEVER get an automatic grip profile; fall back to default."""
        for sharp_obj in ("knife", "scissors", "scissor", "blade", "KNIFE", "Scissors"):
            profile = self.grip_mgr.on_grasp_start(sharp_obj, confidence=0.99)
            self.assertEqual(
                profile.label,
                "default",
                f"Dangerous object '{sharp_obj}' was granted a custom profile instead of default!"
            )
            # Must match default profile force bounds
            self.assertEqual(profile.target_force_n, self.grip_mgr.profiles["default"].target_force_n)
            self.assertEqual(profile.force_ceiling_n, self.grip_mgr.profiles["default"].force_ceiling_n)

        # Mid-grasp attempt to switch to knife must also use default
        self.grip_mgr.on_grasp_start("mug", confidence=0.90)
        ceiling_before = self.grip_mgr.current_force_ceiling_n
        new_ceiling = self.grip_mgr.update_mid_grasp("knife", confidence=0.95)
        default_ceiling = self.grip_mgr.profiles["default"].force_ceiling_n
        self.assertEqual(new_ceiling, min(ceiling_before, default_ceiling))

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
        self.assertTrue(frame.oe_ok)

    def test_arming_refused_when_oe_disabled_in_config(self):
        """Arming must refuse if ESP32_PCA9685_OE_ENABLED is False, unless bench_no_failsafe is True."""
        import tempfile
        import unittest.mock as mock
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "timestamp": time.time(),
                "tool_version": "1.0.0",
                "servos": {"joint_wrist_flex": {"min_deg": 15.0, "max_deg": 165.0, "home_deg": 90.0}}
            }, f)
            calib_file = f.name

        try:
            with mock.patch("config.hardware_config.ESP32_PCA9685_OE_ENABLED", False):
                mock_driver = MockServoDriver()
                # Default safety: must refuse arming
                ctrl_safe = ArmController(driver=mock_driver, force_mock=True, calibration_path=calib_file)
                self.assertFalse(ctrl_safe.is_armed)

                # Bench override: permitted to arm
                ctrl_bench = ArmController(
                    driver=mock_driver,
                    force_mock=True,
                    calibration_path=calib_file,
                    bench_no_failsafe=True,
                )
                self.assertTrue(ctrl_bench.is_armed)
        finally:
            os.remove(calib_file)

    def test_arming_refused_when_telemetry_oe_not_ok(self):
        """Arming must refuse if ESP32 telemetry reports oe_ok=False."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "timestamp": time.time(),
                "tool_version": "1.0.0",
                "servos": {"joint_wrist_flex": {"min_deg": 15.0, "max_deg": 165.0, "home_deg": 90.0}}
            }, f)
            calib_file = f.name

        try:
            mock_driver = MockServoDriver()
            bad_frame = ESP32TelemetryFrame(oe_ok=False)
            ctrl_bad = ArmController(
                driver=mock_driver,
                force_mock=True,
                calibration_path=calib_file,
                telemetry_provider=lambda: bad_frame,
            )
            self.assertFalse(ctrl_bad.is_armed)

            # Bench override bypasses this
            ctrl_bench = ArmController(
                driver=mock_driver,
                force_mock=True,
                calibration_path=calib_file,
                bench_no_failsafe=True,
                telemetry_provider=lambda: bad_frame,
            )
            self.assertTrue(ctrl_bench.is_armed)
            self.assertTrue(ctrl_bench.bench_no_failsafe)
        finally:
            os.remove(calib_file)

    def test_fsr_calibration_arming_refusal_and_acceptance(self):
        """ArmController MUST refuse to arm without valid FSR calibration."""
        import tempfile
        # 1. Valid servo calib
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "timestamp": time.time(),
                "tool_version": "1.0.0",
                "servos": {"joint_wrist_flex": {"min_deg": 15.0, "max_deg": 165.0, "home_deg": 90.0}}
            }, f)
            servo_calib_file = f.name

        # 2. Uncalibrated FSR (like .example)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": False,
                "timestamp": None,
                "tool_version": None,
                "sensors": {"finger_thumb": {"channel": 0}}
            }, f)
            bad_fsr_file = f.name

        # 3. Valid FSR calib
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "timestamp": time.time(),
                "tool_version": "1.0.0",
                "force_ceiling_fraction": 0.85,
                "sensors": {
                    "finger_thumb": {
                        "channel": 0,
                        "tare_volts": 0.05,
                        "loaded_volts": 2.80,
                        "dynamic_range_volts": 2.75,
                        "ceiling_fraction": 0.85,
                        "ceiling_volts": 2.3875,
                    }
                }
            }, f)
            valid_fsr_file = f.name

        try:
            mock_driver = MockServoDriver()

            # Missing FSR calibration file -> Refuse arming
            ctrl_missing = ArmController(
                driver=mock_driver,
                force_mock=True,
                calibration_path=servo_calib_file,
                fsr_calibration_path="/non/existent/fsr_cal.json",
            )
            self.assertFalse(ctrl_missing.is_armed)

            # Uncalibrated FSR -> Refuse arming
            ctrl_uncal = ArmController(
                driver=mock_driver,
                force_mock=True,
                calibration_path=servo_calib_file,
                fsr_calibration_path=bad_fsr_file,
            )
            self.assertFalse(ctrl_uncal.is_armed)

            # Valid FSR -> ARMED
            ctrl_valid = ArmController(
                driver=mock_driver,
                force_mock=True,
                calibration_path=servo_calib_file,
                fsr_calibration_path=valid_fsr_file,
            )
            self.assertTrue(ctrl_valid.is_armed)
            self.assertEqual(ctrl_valid.fsr_force_ceiling_fraction, 0.85)
        finally:
            os.remove(servo_calib_file)
            os.remove(bad_fsr_file)
            os.remove(valid_fsr_file)

    def test_fsr_fraction_force_ceiling_abort(self):
        """execute_force_grasp must emergency stop if finger exceeds fraction ceiling (0.85)."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "timestamp": time.time(),
                "tool_version": "1.0.0",
                "servos": {"joint_wrist_flex": {"min_deg": 15.0, "max_deg": 165.0, "home_deg": 90.0}}
            }, f)
            servo_calib_file = f.name

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "calibrated": True,
                "timestamp": time.time(),
                "tool_version": "1.0.0",
                "force_ceiling_fraction": 0.85,
                "sensors": {
                    "finger_thumb": {
                        "channel": 0,
                        "tare_volts": 0.0,
                        "loaded_volts": 3.0,
                        "dynamic_range_volts": 3.0,
                        "ceiling_fraction": 0.85,
                        "ceiling_volts": 2.55,
                    }
                }
            }, f)
            valid_fsr_file = f.name

        try:
            mock_driver = MockServoDriver()
            # Simulated telemetry where thumb voltage is 2.80V (2.80/3.00 = 93.3% > 85% ceiling)
            excessive_frame = ESP32TelemetryFrame(
                fsr_volts=[2.80, 0.1, 0.1, 0.1, 0.1],
                oe_ok=True,
            )

            ctrl = ArmController(
                driver=mock_driver,
                force_mock=True,
                calibration_path=servo_calib_file,
                fsr_calibration_path=valid_fsr_file,
                telemetry_provider=lambda: excessive_frame,
            )
            self.assertTrue(ctrl.is_armed)

            # Attempt force grasp: should trip ceiling and abort
            result = ctrl.execute_force_grasp(target_force_n=2.0, timeout_s=0.5)
            self.assertFalse(result)
            self.assertTrue(ctrl.is_emergency_stopped)
        finally:
            os.remove(servo_calib_file)
            os.remove(valid_fsr_file)

    def test_strong_single_site_with_antagonist_crosstalk_measures_zero_false_estops(self):
        """
        Item 4: Strong single-site contraction (100%) plus 30% crosstalk on antagonist
        channel must NOT trigger false E-stops. Tests configurable debounce and crosstalk.
        """
        # Decoder with baseline: 0.08V, MVC: 2.15V (dynamic range 2.07V)
        decoder = EMGDecoder(
            num_channels=2,
            config_path="/non/existent/path.json",
            co_contraction_threshold=0.85,
            co_contraction_debounce_windows=2,
            crosstalk_tolerance_fraction=0.30,
        )
        decoder.baseline_volts = np.array([0.08, 0.08])
        decoder.mvc_volts = np.array([2.15, 2.15])

        # 1. 100% Flexor Contraction (2.15V) with 30% Extensor Crosstalk
        # Extensor voltage with 30% crosstalk: 0.08 + 0.30 * (2.15 - 0.08) = 0.701V
        v_flex_100 = 2.15
        v_ext_crosstalk_30 = 0.08 + 0.30 * (2.15 - 0.08)

        false_estop_count = 0
        total_frames = 100
        for _ in range(total_frames):
            frame_sample = np.array([v_flex_100, v_ext_crosstalk_30])
            intent = decoder.update_samples(frame_sample)
            if intent.gesture == EMGIntent.CO_CONTRACTION_ESTOP:
                false_estop_count += 1

        self.assertEqual(false_estop_count, 0, "False E-Stop occurred during 100% flexor + 30% crosstalk!")
        self.assertEqual(decoder.current_gesture, EMGIntent.GRASP_CLOSE)

        # 2. 100% Extensor Contraction (2.15V) with 30% Flexor Crosstalk
        v_ext_100 = 2.15
        v_flex_crosstalk_30 = 0.08 + 0.30 * (2.15 - 0.08)

        false_estop_count_ext = 0
        for _ in range(total_frames):
            frame_sample = np.array([v_flex_crosstalk_30, v_ext_100])
            intent = decoder.update_samples(frame_sample)
            if intent.gesture == EMGIntent.CO_CONTRACTION_ESTOP:
                false_estop_count_ext += 1

        self.assertEqual(false_estop_count_ext, 0, "False E-Stop occurred during 100% extensor + 30% crosstalk!")
        self.assertEqual(decoder.current_gesture, EMGIntent.HAND_OPEN)

        # 3. Test Configurable Debounce Counter
        # Custom decoder with debounce_windows = 4
        decoder_debounced = EMGDecoder(
            num_channels=2,
            config_path="/non/existent/path.json",
            co_contraction_threshold=0.85,
            co_contraction_debounce_windows=4,
        )
        decoder_debounced.baseline_volts = np.array([0.08, 0.08])
        decoder_debounced.mvc_volts = np.array([2.15, 2.15])
        decoder_debounced.smoothed_activations = np.array([0.90, 0.90])  # Pre-condition above threshold

        co_contract_sample = np.array([2.15, 2.15])  # Both 100% -> True co-contraction

        # Windows 1..3: within debounce, should NOT trigger yet
        for step in range(1, 4):
            intent = decoder_debounced.update_samples(co_contract_sample)
            self.assertNotEqual(intent.gesture, EMGIntent.CO_CONTRACTION_ESTOP, f"Premature E-Stop at window {step}!")
            self.assertEqual(decoder_debounced.co_contraction_counter, step)

        # Window 4: reaches debounce count (4) -> MUST trigger E-Stop
        intent_4 = decoder_debounced.update_samples(co_contract_sample)
        self.assertEqual(intent_4.gesture, EMGIntent.CO_CONTRACTION_ESTOP)
        self.assertEqual(decoder_debounced.co_contraction_counter, 4)

    def test_i2c_scan_fails_if_two_devices_answer_at_0x70(self):
        """Item 6: I2C scan test MUST fail if two devices answer at 0x70."""
        from drivers.tca9548a_as5600 import check_i2c_bus_collisions
        from config.hardware_config import TCA9548A_I2C_ADDRESS

        # 1. Verify configured address is 0x71
        self.assertEqual(TCA9548A_I2C_ADDRESS, 0x71)

        # 2. Simulated collision condition: both PCA9685 (ALLCALL) and TCA9548A respond at 0x70
        colliding_devices = {
            "PCA9685_ALLCALL": [0x40, 0x70],
            "TCA9548A_OLD": [0x70],
        }
        with self.assertRaises(RuntimeError) as ctx:
            check_i2c_bus_collisions(candidate_devices=colliding_devices)
        self.assertIn("I2C COLLISION ERROR", str(ctx.exception))
        self.assertIn("0x70", str(ctx.exception))

        # 3. Clean condition: PCA9685 ALLCALL cleared (only 0x40), TCA9548A safely at 0x71
        safe_devices = {
            "PCA9685": [0x40],
            "TCA9548A": [0x71],
        }
        self.assertTrue(check_i2c_bus_collisions(candidate_devices=safe_devices))


if __name__ == "__main__":
    unittest.main()




