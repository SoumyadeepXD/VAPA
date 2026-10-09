"""
VAPA Master Orchestrator Engine
Multi-threaded executive engine integrating 3D RealSense vision,
ESP32 biosignals & 5-finger FSR tactile sensors, Inverse Kinematics,
9-servo PCA9685 hardware actuation, and comprehensive real-time HUD dashboard.
"""

import time
import math
import logging
import threading
import numpy as np
import cv2

from config.system_config import (
    MAIN_LOOP_RATE_HZ,
    VISION_PROCESS_RATE_HZ,
    BIOSIGNAL_PROCESS_RATE_HZ,
    REACH_TIMEOUT_S,
    GRASP_TIMEOUT_S,
    EMG_DECODER,
    EMG_CLASSIFIER_CONFIDENCE_THRESHOLD,
    EMG_CLASSIFIER_MAJORITY_VOTING_N,
    EMG_CLASSIFIER_WINDOW_MS,
)
from vision.realsense_camera import RealSenseCamera
from vision.object_detector import ObjectDetector
from vision.spatial_3d import Spatial3DAnalyzer, GraspTarget3D
from vision.visualizer_3d import VisionVisualizer
from biosignals.biosignal_streamer import BiosignalStreamer
from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.emg_classifier import EMGClassifier
from biosignals.eeg_decoder import EEGDecoder, EEGIntent
from biosignals.intent_fusion import IntentFusionEngine, MultimodalCommand
from kinematics.arm_model import ArmModel
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import InverseKinematics
from kinematics.trajectory_planner import TrajectoryPlanner
from actuation.arm_controller import ArmController
from actuation.pca9685_controller import PCA9685ServoDriver
from drivers.tca9548a_as5600 import AS5600EncoderMux
from core.state_machine import VAPAStateMachine, VAPAState

logger = logging.getLogger("VAPA.Engine")


class VAPAEngine:
    """Master multi-threaded control engine for Visually-Assisted Prosthetic Arm."""
    def __init__(self, force_mock: bool = False, bench_no_failsafe: bool = False):
        self.force_mock = force_mock
        self.bench_no_failsafe = bench_no_failsafe
        self.running = False
        self.lock = threading.Lock()

        # Initialize Subsystems
        logger.info("========================================================")
        logger.info("Initializing VAPA subsystems on NVIDIA Jetson Orin...")
        logger.info("========================================================")

        # 1. Vision Subsystem (Intel RealSense D435/D455)
        self.camera = RealSenseCamera(force_mock=force_mock)
        self.detector = ObjectDetector()
        self.spatial = Spatial3DAnalyzer()
        self.visualizer = VisionVisualizer()

        # 2. Biosignal & Tactile Subsystem (ESP32 UART Telemetry)
        self.streamer = BiosignalStreamer(force_mock=force_mock)
        if EMG_DECODER == "classifier":
            logger.info("Initializing EMGClassifier (Safe ML architecture with fail-closed fallback)...")
            self.emg_decoder = EMGClassifier(
                confidence_threshold=EMG_CLASSIFIER_CONFIDENCE_THRESHOLD,
                majority_vote_window=EMG_CLASSIFIER_MAJORITY_VOTING_N,
                window_duration_ms=EMG_CLASSIFIER_WINDOW_MS,
            )
        else:
            logger.info("Initializing baseline EMGDecoder (Dual-threshold envelope architecture)...")
            self.emg_decoder = EMGDecoder()
        self.eeg_decoder = EEGDecoder()
        self.fusion = IntentFusionEngine()

        # 3. Kinematics & Actuation Subsystem (PCA9685 9-Servo Arm & Hand)
        self.arm_model = ArmModel()
        self.fk = ForwardKinematics(self.arm_model)
        self.ik = InverseKinematics(self.arm_model)
        self.planner = TrajectoryPlanner()
        self.arm = ArmController(
            force_mock=force_mock,
            bench_no_failsafe=bench_no_failsafe,
            telemetry_provider=self.streamer.get_latest_telemetry,
        )
        self.encoders = AS5600EncoderMux(force_mock=force_mock)

        # 4. State Machine
        self.state_machine = VAPAStateMachine()

        # Log Hardware Status Diagnostic Banner
        self._log_subsystem_diagnostics()

        # Shared Real-Time State Data
        self.latest_color = None
        self.latest_depth = None
        self.visible_targets: list[GraspTarget3D] = []
        self.selected_target_idx = 0
        self.latest_emg_intent = EMGIntent(EMGIntent.REST, 1.0, 0.0, 0.0, [0.0]*4, time.time())
        self.latest_eeg_intent = EEGIntent(EEGIntent.IDLE, 1.0, 0.35, 0.20, 0.5, False, time.time())
        self.latest_multimodal_cmd = MultimodalCommand(MultimodalCommand.NO_OP)
        self.latest_fsr_forces = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.current_fps = 0.0

        # Background Threads
        self.vision_thread = None
        self.biosignal_thread = None
        self.control_thread = None

        logger.info("VAPA Engine successfully initialized.")

    def _log_subsystem_diagnostics(self):
        cam_status = "[ONLINE: RealSense USB3]" if not self.camera.is_synthetic else "[FALLBACK: Mock Scene]"
        esp_status = f"[ONLINE: {self.streamer.esp32_receiver.port}]" if self.streamer.is_connected else "[FALLBACK: Mock UART]"
        pca_driver = getattr(self.arm, "driver", None)
        pca_connected = getattr(pca_driver, "is_connected", False)
        pca_bus = getattr(pca_driver, "bus_num", "N/A")
        pca_status = f"[ONLINE: /dev/i2c-{pca_bus}]" if pca_connected else "[FALLBACK: Mock Servos]"
        enc_status = f"[ONLINE: /dev/i2c-{self.encoders.bus_num}]" if self.encoders.is_connected else "[FALLBACK: Mock Encoders]"

        logger.info("---------------- HARDWARE STATUS SUMMARY ----------------")
        logger.info(f"  1. 3D Camera       : {cam_status}")
        logger.info(f"  2. ESP32 UART Node : {esp_status}")
        logger.info(f"  3. PCA9685 Servos  : {pca_status}")
        logger.info(f"  4. TCA9548A Encoders: {enc_status}")
        logger.info("---------------------------------------------------------")

    def start(self):
        """Starts all concurrent subsystem processing loops."""
        self.running = True
        self.vision_thread = threading.Thread(target=self._vision_loop, name="VisionThread", daemon=True)
        self.biosignal_thread = threading.Thread(target=self._biosignal_loop, name="BiosignalThread", daemon=True)
        self.control_thread = threading.Thread(target=self._control_loop, name="ControlThread", daemon=True)

        self.vision_thread.start()
        self.biosignal_thread.start()
        self.control_thread.start()
        logger.info("VAPA Engine background threads running.")

    def stop(self):
        """Gracefully shuts down all threads and releases hardware."""
        self.running = False
        time.sleep(0.1)
        self.arm.go_to_home(duration_s=0.5)
        self.camera.stop()
        self.streamer.close()
        self.arm.close()
        self.encoders.close()
        logger.info("VAPA Engine stopped.")

    # ==========================================================================
    # 1. VISION THREAD (30 Hz)
    # ==========================================================================
    def _vision_loop(self):
        dt = 1.0 / VISION_PROCESS_RATE_HZ
        last_time = time.time()

        while self.running:
            start_tick = time.time()

            color, depth = self.camera.get_frames()
            if color is not None and depth is not None:
                # Run 3D Object Detection
                detections = self.detector.detect(color, depth_image_m=depth)
                intrinsics_dict = self.camera.get_intrinsics_dict()
                targets = self.spatial.analyze_scene(detections, color, depth, intrinsics_dict)

                with self.lock:
                    self.latest_color = color
                    self.latest_depth = depth
                    self.visible_targets = targets

            # Compute FPS
            now = time.time()
            self.current_fps = 1.0 / max(1e-4, now - last_time)
            last_time = now

            elapsed = time.time() - start_tick
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ==========================================================================
    # 2. BIOSIGNAL & TACTILE THREAD (100 Hz)
    # ==========================================================================
    def _biosignal_loop(self):
        dt = 1.0 / BIOSIGNAL_PROCESS_RATE_HZ

        while self.running:
            start_tick = time.time()

            # Read biosignal chunk from ESP32 UART
            emg_chunk, eeg_chunk = self.streamer.read_chunk(num_samples=10)
            fsr_forces = self.streamer.get_fsr_forces()
            telemetry = self.streamer.get_latest_telemetry()
            hw_estop = getattr(telemetry, "estop_button_pressed", False)

            emg_intent = self.emg_decoder.update_samples(emg_chunk)
            eeg_intent = self.eeg_decoder.update_samples(eeg_chunk)

            with self.lock:
                targets_copy = list(self.visible_targets)
                cur_state = self.state_machine.current_state

            cmd = self.fusion.fuse(emg_intent, eeg_intent, targets_copy, cur_state, estop_hardware_trigger=hw_estop)

            # High-priority instant emergency stop directly from biosignals thread or hardware button
            if cmd.action == MultimodalCommand.EMERGENCY_STOP or hw_estop:
                self.state_machine.transition_to(VAPAState.EMERGENCY_STOP)
                self.arm.emergency_stop()

            with self.lock:
                self.latest_emg_intent = emg_intent
                self.latest_eeg_intent = eeg_intent
                self.latest_multimodal_cmd = cmd
                self.latest_fsr_forces = fsr_forces
                self.selected_target_idx = self.fusion.selected_target_index

            elapsed = time.time() - start_tick
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ==========================================================================
    # 3. MASTER CONTROL & KINEMATICS LOOP (50 Hz)
    # ==========================================================================
    def _control_loop(self):
        dt = 1.0 / MAIN_LOOP_RATE_HZ

        while self.running:
            start_tick = time.time()

            with self.lock:
                cur_state = self.state_machine.current_state
                cmd = self.latest_multimodal_cmd
                targets = list(self.visible_targets)
                target_idx = self.selected_target_idx

            # 1. Emergency Stop Check
            if cmd.action == MultimodalCommand.EMERGENCY_STOP or self.arm.is_emergency_stopped:
                if self.state_machine.current_state != VAPAState.EMERGENCY_STOP:
                    self.state_machine.transition_to(VAPAState.EMERGENCY_STOP)
                    self.arm.emergency_stop()

            # 2. State Machine Logic:
            # --- IDLE / SCANNING ---
            if cur_state in (VAPAState.IDLE, VAPAState.SCANNING):
                if cmd.action == MultimodalCommand.START_REACH and cmd.target is not None:
                    target = cmd.target
                    logger.info(f"Target selected: {target.label} at Base coords {target.center_base_m}")
                    self.state_machine.transition_to(VAPAState.PLANNING, target=target)

            # --- PLANNING ---
            elif cur_state == VAPAState.PLANNING:
                target = self.state_machine.locked_target
                if target is None or not target.is_reachable:
                    logger.warning("Target invalid or unreachable. Returning to SCANNING.")
                    self.state_machine.transition_to(VAPAState.SCANNING)
                else:
                    approach_offset = -0.04 * target.approach_vector
                    reach_pos = target.center_base_m + approach_offset

                    target_angles, success = self.ik.solve_analytical(
                        reach_pos,
                        target_pitch_deg=target.grasp_pitch_deg,
                        target_roll_deg=0.0,
                        gripper_percent=100.0,
                    )

                    if success:
                        current_angles = self.arm.get_joint_angles()
                        trajectory = self.planner.plan_trajectory(current_angles, target_angles, min_duration_s=1.2)
                        self.state_machine.transition_to(VAPAState.REACHING, target=target)

                        reach_ok = self.arm.execute_trajectory(
                            trajectory,
                            abort_check_fn=lambda: (
                                self.state_machine.current_state == VAPAState.EMERGENCY_STOP
                                or self.arm.is_emergency_stopped
                            ),
                        )
                        if reach_ok:
                            self.state_machine.transition_to(VAPAState.AT_TARGET, target=target)
                        else:
                            self.state_machine.transition_to(VAPAState.IDLE)
                    else:
                        logger.warning(f"IK solver failed for target position {reach_pos}. Aborting reach.")
                        self.state_machine.transition_to(VAPAState.SCANNING)

            # --- AT TARGET / GRASPING ---
            elif cur_state in (VAPAState.AT_TARGET, VAPAState.GRASPING):
                target = self.state_machine.locked_target
                if cmd.action == MultimodalCommand.START_GRASP:
                    self.state_machine.transition_to(VAPAState.GRASPING, target=target)
                    grasp_force = target.target_force_n if target else cmd.target_force_n

                    # Smoothly advance fingers onto target if pre-grasp standoff was used
                    if target is not None:
                        advance_angles, adv_ok = self.ik.solve_analytical(
                            target.center_base_m,
                            target_pitch_deg=target.grasp_pitch_deg,
                            target_roll_deg=0.0,
                            gripper_percent=100.0,
                        )
                        if adv_ok:
                            cur_q = self.arm.get_joint_angles()
                            adv_traj = self.planner.plan_trajectory(cur_q, advance_angles, min_duration_s=0.5)
                            self.arm.execute_trajectory(
                                adv_traj,
                                abort_check_fn=lambda: (
                                    self.state_machine.current_state == VAPAState.EMERGENCY_STOP
                                    or self.arm.is_emergency_stopped
                                ),
                            )

                    # Execute closed-loop grasp with live FSR feedback
                    grasp_success = self.arm.execute_force_grasp(
                        target_force_n=grasp_force,
                        timeout_s=GRASP_TIMEOUT_S,
                        tactile_sensor_fn=self.streamer.get_total_grip_force_n,
                    )

                    if grasp_success:
                        self.state_machine.transition_to(VAPAState.HOLDING, target=target)
                    else:
                        logger.warning("Grasp did not secure target. Opening fingers and returning home.")
                        self.arm.release_grasp()
                        self.arm.go_to_home(duration_s=1.2)
                        self.state_machine.transition_to(VAPAState.IDLE)

            # --- HOLDING ---
            elif cur_state == VAPAState.HOLDING:
                if cmd.action == MultimodalCommand.RELEASE_GRIP:
                    self.state_machine.transition_to(VAPAState.RELEASING)
                    self.arm.release_grasp()
                    time.sleep(0.3)
                    self.arm.go_to_home(duration_s=1.5)
                    self.state_machine.transition_to(VAPAState.IDLE)

            # --- EMERGENCY STOP ---
            elif cur_state == VAPAState.EMERGENCY_STOP:
                pass

            elapsed = time.time() - start_tick
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ==========================================================================
    # 4. COMPOSITE HUD DASHBOARD GENERATOR
    # ==========================================================================
    def get_dashboard_frame(self) -> np.ndarray:
        """Constructs unified multi-view HUD frame with live tactile, neural, and servo telemetry."""
        with self.lock:
            color = self.latest_color
            depth = self.latest_depth
            targets = list(self.visible_targets)
            target_idx = self.selected_target_idx
            emg = self.latest_emg_intent
            eeg = self.latest_eeg_intent
            fsr_forces = list(self.latest_fsr_forces)
            state = self.state_machine.current_state
            fps = self.current_fps
            joint_angles = self.arm.get_joint_angles()

        if color is None or depth is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Initializing VAPA Hardware...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return blank

        # 1. Vision HUD (RGB stream with 3D overlays)
        vis_rgb = self.visualizer.draw_scene(color, targets, selected_index=target_idx, fps=fps, system_state=state)

        # 2. Depth Colormap
        depth_color = self.visualizer.render_depth_colormap(depth, max_depth_m=1.5)

        # 3. Telemetry Panel (Height: 240, Width: 320)
        telemetry_panel = np.zeros((240, 320, 3), dtype=np.uint8)
        telemetry_panel[:] = (25, 25, 25)

        # Panel Header & Status
        cv2.rectangle(telemetry_panel, (0, 0), (320, 22), (15, 15, 15), -1)
        cv2.putText(telemetry_panel, "VAPA HARDWARE TELEMETRY", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1)

        # Hardware Links Status Badges
        y_pos = 34
        cam_hw = "REAL" if not self.camera.is_synthetic else "MOCK"
        esp_hw = "REAL" if self.streamer.is_connected else "MOCK"
        pca_driver = getattr(self.arm, "driver", None)
        pca_hw = "REAL" if getattr(pca_driver, "is_connected", False) else "MOCK"

        c_green = (0, 220, 0)
        c_orange = (0, 160, 255)
        cv2.putText(telemetry_panel, f"CAM:{cam_hw}", (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, c_green if cam_hw == "REAL" else c_orange, 1)
        cv2.putText(telemetry_panel, f"ESP32:{esp_hw}", (95, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, c_green if esp_hw == "REAL" else c_orange, 1)
        cv2.putText(telemetry_panel, f"PCA:{pca_hw}", (195, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, c_green if pca_hw == "REAL" else c_orange, 1)

        # FSR 402 Fingertip Tactile Sensors (5 fingers)
        cv2.line(telemetry_panel, (8, 42), (312, 42), (55, 55, 55), 1)
        f_names = ["Th", "In", "Mi", "Ri", "Li"]
        total_f = sum(fsr_forces)
        cv2.putText(telemetry_panel, f"FSR FORCE (N) | Total: {total_f:.1f} N", (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)

        for i in range(5):
            f_val = fsr_forces[i] if i < len(fsr_forces) else 0.0
            x_bar = 8 + i * 62
            cv2.putText(telemetry_panel, f"{f_names[i]}:{f_val:.1f}", (x_bar, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)
            bar_h = int(np.clip((f_val / 6.0) * 10.0, 0, 10))
            cv2.rectangle(telemetry_panel, (x_bar, 70), (x_bar + 48, 80), (50, 50, 50), -1)
            if bar_h > 0:
                cv2.rectangle(telemetry_panel, (x_bar, 80 - bar_h), (x_bar + 48, 80), (0, 255, 255), -1)

        # EMG Section (MyoWare 2.0 on ADS1115 A0)
        cv2.line(telemetry_panel, (8, 86), (312, 86), (55, 55, 55), 1)
        emg_col = (0, 255, 0) if emg.gesture == EMGIntent.GRASP_CLOSE else ((0, 200, 255) if emg.gesture == EMGIntent.HAND_OPEN else (200, 200, 200))
        cv2.putText(telemetry_panel, f"EMG: {emg.gesture} ({emg.confidence*100:.0f}%)", (8, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.34, emg_col, 1)
        cv2.rectangle(telemetry_panel, (8, 102), (312, 110), (50, 50, 50), -1)
        act_w = int(304 * np.clip(emg.activation_level, 0.0, 1.0))
        cv2.rectangle(telemetry_panel, (8, 102), (8 + act_w, 110), (0, 220, 0) if emg.activation_level < 0.85 else (0, 0, 255), -1)

        # EEG Section
        cv2.putText(telemetry_panel, f"EEG MI: {'ACTIVE' if eeg.motor_imagery_active else 'IDLE'} | Attn: {eeg.attention_score:.2f}", (8, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 180, 50), 1)

        # 9-Servo Positions (Fingers 0-4, Wrist 5-7, Forearm 8)
        cv2.line(telemetry_panel, (8, 128), (312, 128), (55, 55, 55), 1)
        servo_items = [
            ("CH0 Th", joint_angles.get("finger_thumb", 0.0)),
            ("CH1 In", joint_angles.get("finger_index", 0.0)),
            ("CH2 Mi", joint_angles.get("finger_middle", 0.0)),
            ("CH3 Ri", joint_angles.get("finger_ring", 0.0)),
            ("CH4 Pi", joint_angles.get("finger_pinky", 0.0)),
            ("CH5 W-Flx", joint_angles.get("joint_wrist_flex", 90.0)),
            ("CH6 W-Rot", joint_angles.get("joint_wrist_rotate", 90.0)),
            ("CH7 W-Bnd", joint_angles.get("joint_wrist_bend", 90.0)),
            ("CH8 F-Rot", joint_angles.get("joint_forearm_rotate", 90.0)),
        ]
        y_s = 140
        for i, (s_label, s_ang) in enumerate(servo_items):
            col_x = 8 if i < 5 else 165
            row_y = y_s + (i if i < 5 else i - 5) * 12
            cv2.putText(telemetry_panel, f"{s_label}:", (col_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1)
            cv2.putText(telemetry_panel, f"{s_ang:5.1f}*", (col_x + 80, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1)

        # Target Info Section
        cv2.line(telemetry_panel, (8, 204), (312, 204), (55, 55, 55), 1)
        if targets and target_idx < len(targets):
            t_sel = targets[target_idx]
            cb = t_sel.center_base_m
            cv2.putText(telemetry_panel, f"[{target_idx+1}] {t_sel.label.upper()} ({t_sel.score:.2f}) | F:{t_sel.target_force_n:.1f}N", (8, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 0), 1)
            cv2.putText(telemetry_panel, f"Pos: [{cb[0]:.2f}, {cb[1]:.2f}, {cb[2]:.2f}]m | [TAB] Cycle [G] Grasp", (8, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1)
        else:
            cv2.putText(telemetry_panel, "Scanning room... | [TAB] Cycle | [G] Grasp", (8, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

        # Depth thumbnail and composite HUD
        depth_small = cv2.resize(depth_color, (320, 240))
        right_column = np.vstack((depth_small, telemetry_panel))

        composite = np.hstack((vis_rgb, right_column))
        return composite

    def trigger_gesture(self, gesture_name: str, duration_s: float = 1.0):
        self.streamer.trigger_synthetic_gesture(gesture_name, duration_s)

    def cycle_target(self):
        if self.visible_targets:
            self.selected_target_idx = (self.selected_target_idx + 1) % len(self.visible_targets)
            self.fusion.selected_target_index = self.selected_target_idx

    def reset_estop(self):
        self.arm.reset_emergency_stop()
        self.state_machine.reset_from_estop()
