"""
VAPA Master Orchestrator Engine
Multi-threaded executive engine integrating 3D RealSense vision,
EEG/EMG biosignal decoders, Inverse Kinematics, multi-servo actuation, and HUD.
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
    WORKSPACE_BOUNDS_M,
)
from vision.realsense_camera import RealSenseCamera
from vision.object_detector import ObjectDetector
from vision.spatial_3d import Spatial3DAnalyzer, GraspTarget3D
from vision.visualizer_3d import VisionVisualizer
from biosignals.biosignal_streamer import BiosignalStreamer
from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.eeg_decoder import EEGDecoder, EEGIntent
from biosignals.intent_fusion import IntentFusionEngine, MultimodalCommand
from kinematics.arm_model import ArmModel
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import InverseKinematics
from kinematics.trajectory_planner import TrajectoryPlanner
from actuation.arm_controller import ArmController
from core.state_machine import VAPAStateMachine, VAPAState

logger = logging.getLogger("VAPA.Engine")


class VAPAEngine:
    """Master multi-threaded control engine for Visually-Assisted Prosthetic Arm."""
    def __init__(self, force_mock: bool = False):
        self.force_mock = force_mock
        self.running = False
        self.lock = threading.Lock()

        # Initialize Subsystems
        logger.info("Initializing VAPA subsystems...")
        self.camera = RealSenseCamera(force_mock=force_mock)
        self.detector = ObjectDetector()
        self.spatial = Spatial3DAnalyzer()
        self.visualizer = VisionVisualizer()

        self.streamer = BiosignalStreamer(force_mock=force_mock)
        self.emg_decoder = EMGDecoder()
        self.eeg_decoder = EEGDecoder()
        self.fusion = IntentFusionEngine()

        self.arm_model = ArmModel()
        self.fk = ForwardKinematics(self.arm_model)
        self.ik = InverseKinematics(self.arm_model)
        self.planner = TrajectoryPlanner()
        self.arm = ArmController(force_mock=force_mock)

        self.state_machine = VAPAStateMachine()

        # Shared Real-Time State Data
        self.latest_color = None
        self.latest_depth = None
        self.visible_targets: list[GraspTarget3D] = []
        self.selected_target_idx = 0
        self.latest_emg_intent = EMGIntent(EMGIntent.REST, 1.0, 0.0, 0.0, [0.0]*4, time.time())
        self.latest_eeg_intent = EEGIntent(EEGIntent.IDLE, 1.0, 0.35, 0.20, 0.5, False, time.time())
        self.latest_multimodal_cmd = MultimodalCommand(MultimodalCommand.NO_OP)
        self.current_fps = 0.0

        # Background Threads
        self.vision_thread = None
        self.biosignal_thread = None
        self.control_thread = None

        logger.info("VAPA Engine successfully initialized.")

    def start(self):
        """Starts all concurrent subsystem processing loops."""
        self.running = True
        self.vision_thread = threading.Thread(target=self._vision_loop, name="VisionThread", daemon=True)
        self.biosignal_thread = threading.Thread(target=self._biosignal_loop, name="BiosignalThread", daemon=True)
        self.control_thread = threading.Thread(target=self._control_loop, name="ControlThread", daemon=True)

        self.vision_thread.start()
        self.biosignal_thread.start()
        self.control_thread.start()
        logger.info("VAPA Engine threads started.")

    def stop(self):
        """Gracefully shuts down all threads and releases hardware."""
        self.running = False
        time.sleep(0.1)
        self.arm.go_to_home(duration_s=0.5)
        self.arm.close()
        self.camera.stop()
        self.streamer.close()
        logger.info("VAPA Engine stopped.")

    # ==========================================================================
    # 1. VISION THREAD (15 - 30 Hz)
    # ==========================================================================
    def _vision_loop(self):
        dt = 1.0 / VISION_PROCESS_RATE_HZ
        last_time = time.time()

        while self.running:
            start_tick = time.time()
            color, depth = self.camera.get_frames()

            if color is not None and depth is not None:
                detections = self.detector.detect(color, depth)
                targets = self.spatial.process_scene(detections, color, depth, self.camera)

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
    # 2. BIOSIGNAL THREAD (100 Hz)
    # ==========================================================================
    def _biosignal_loop(self):
        dt = 1.0 / BIOSIGNAL_PROCESS_RATE_HZ

        while self.running:
            start_tick = time.time()
            emg_chunk, eeg_chunk = self.streamer.read_chunk(num_samples=10)

            emg_intent = self.emg_decoder.update_samples(emg_chunk)
            eeg_intent = self.eeg_decoder.update_samples(eeg_chunk)

            with self.lock:
                targets_copy = list(self.visible_targets)
                cur_state = self.state_machine.current_state

            cmd = self.fusion.fuse(emg_intent, eeg_intent, targets_copy, cur_state)

            with self.lock:
                self.latest_emg_intent = emg_intent
                self.latest_eeg_intent = eeg_intent
                self.latest_multimodal_cmd = cmd
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
            if cmd.action == MultimodalCommand.EMERGENCY_STOP:
                self.state_machine.transition_to(VAPAState.EMERGENCY_STOP)
                self.arm.emergency_stop()

            # 2. STATE MACHINE LOGIC:

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
                    # Pre-approach position (5cm offset along approach vector to prevent collision)
                    approach_offset = -0.04 * target.approach_vector
                    reach_pos = target.center_base_m + approach_offset

                    # Solve IK
                    target_angles, success = self.ik.solve_analytical(
                        reach_pos,
                        target_pitch_deg=target.grasp_pitch_deg,
                        target_roll_deg=0.0,
                        gripper_percent=100.0,  # Open gripper fully during reach
                    )

                    if success:
                        current_angles = self.arm.get_joint_angles()
                        trajectory = self.planner.plan_trajectory(current_angles, target_angles, min_duration_s=1.2)
                        self.state_machine.transition_to(VAPAState.REACHING, target=target)

                        # Execute reach trajectory
                        reach_ok = self.arm.execute_trajectory(
                            trajectory,
                            abort_check_fn=lambda: self.state_machine.current_state == VAPAState.EMERGENCY_STOP,
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
                    grasp_success = self.arm.execute_force_grasp(target_force_n=grasp_force, timeout_s=GRASP_TIMEOUT_S)

                    if grasp_success:
                        self.state_machine.transition_to(VAPAState.HOLDING, target=target)
                    else:
                        logger.warning("Grasp did not secure target. Opening gripper and returning home.")
                        self.arm.release_grasp()
                        self.arm.go_to_home(duration_s=1.2)
                        self.state_machine.transition_to(VAPAState.IDLE)

            # --- HOLDING ---
            elif cur_state == VAPAState.HOLDING:
                # Modulate force if requested
                if cmd.action == MultimodalCommand.MODULATE_FORCE:
                    pass
                elif cmd.action == MultimodalCommand.RELEASE_GRIP:
                    self.state_machine.transition_to(VAPAState.RELEASING)
                    self.arm.release_grasp()
                    time.sleep(0.3)
                    self.arm.go_to_home(duration_s=1.5)
                    self.state_machine.transition_to(VAPAState.IDLE)

            # --- EMERGENCY STOP ---
            elif cur_state == VAPAState.EMERGENCY_STOP:
                # Can be reset via manual key or reset call
                pass

            elapsed = time.time() - start_tick
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ==========================================================================
    # 4. COMPOSITE HUD DASHBOARD GENERATOR
    # ==========================================================================
    def get_dashboard_frame(self) -> np.ndarray:
        """Constructs unified multi-view HUD frame (RGB + 3D Bounding + Depth + EMG/EEG Waveforms + Kinematics)."""
        with self.lock:
            color = self.latest_color
            depth = self.latest_depth
            targets = list(self.visible_targets)
            target_idx = self.selected_target_idx
            emg = self.latest_emg_intent
            eeg = self.latest_eeg_intent
            state = self.state_machine.current_state
            fps = self.current_fps
            joint_angles = self.arm.get_joint_angles()

        if color is None or depth is None:
            # Placeholder frame if camera warming up
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Initializing VAPA Sensors...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return blank

        # 1. Vision HUD (RGB stream with 3D overlays)
        vis_rgb = self.visualizer.draw_scene(color, targets, selected_index=target_idx, fps=fps, system_state=state)

        # 2. Depth Colormap
        depth_color = self.visualizer.render_depth_colormap(depth, max_depth_m=1.5)

        # 3. Biosignal & Arm Telemetry Panel (Height: 480, Width: 320)
        telemetry_panel = np.zeros((480, 320, 3), dtype=np.uint8)
        telemetry_panel[:] = (30, 30, 30)

        # Panel Header
        cv2.rectangle(telemetry_panel, (0, 0), (320, 36), (15, 15, 15), -1)
        cv2.putText(telemetry_panel, "NEURAL & ARM TELEMETRY", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 2)

        # EMG Section
        cv2.putText(telemetry_panel, f"EMG MUSCLE STATE:", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        emg_color = (0, 255, 0) if emg.gesture == EMGIntent.GRASP_CLOSE else ((0, 200, 255) if emg.gesture == EMGIntent.HAND_OPEN else (200, 200, 200))
        cv2.putText(telemetry_panel, f"Intent: {emg.gesture} ({emg.confidence*100:.0f}%)", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, emg_color, 2)
        cv2.putText(telemetry_panel, f"Proportional Force: {emg.proportional_force_n:.1f} N", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 255), 1)

        # EMG Activation Level Bar
        cv2.rectangle(telemetry_panel, (12, 112), (308, 126), (60, 60, 60), -1)
        bar_w = int(296 * emg.activation_level)
        cv2.rectangle(telemetry_panel, (12, 112), (12 + bar_w, 126), (0, 220, 0) if emg.activation_level < 0.8 else (0, 0, 255), -1)

        # EEG Section
        cv2.line(telemetry_panel, (12, 140), (308, 140), (80, 80, 80), 1)
        cv2.putText(telemetry_panel, f"EEG BRAIN WAVE STATE:", (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(telemetry_panel, f"Command: {eeg.command}", (12, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 50), 2)
        cv2.putText(telemetry_panel, f"Motor Imagery: {'ACTIVE' if eeg.motor_imagery_active else 'IDLE'} | Mu: {eeg.mu_power:.2f}", (12, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        cv2.putText(telemetry_panel, f"Attention Score: {eeg.attention_score:.2f}", (12, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        # Arm Joint Angles Section
        cv2.line(telemetry_panel, (12, 235), (308, 235), (80, 80, 80), 1)
        cv2.putText(telemetry_panel, "ARM JOINT POSITIONS (DEG):", (12, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        y_offset = 276
        for j_idx, (j_name, angle) in enumerate(joint_angles.items()):
            short_name = j_name.replace("joint_", "J").replace("_", " ")
            cv2.putText(telemetry_panel, f"{short_name[:14]}:", (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            cv2.putText(telemetry_panel, f"{angle:6.1f}", (240, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            y_offset += 18

        # Target Info Section
        cv2.line(telemetry_panel, (12, 395), (308, 395), (80, 80, 80), 1)
        cv2.putText(telemetry_panel, "ACTIVE TARGET:", (12, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if targets and target_idx < len(targets):
            t_sel = targets[target_idx]
            cb = t_sel.center_base_m
            cv2.putText(telemetry_panel, f"[{target_idx+1}] {t_sel.label.upper()} ({t_sel.score:.2f})", (12, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.putText(telemetry_panel, f"Base: X:{cb[0]:.2f} Y:{cb[1]:.2f} Z:{cb[2]:.2f}m", (12, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            cv2.putText(telemetry_panel, f"Grasp Force: {t_sel.target_force_n:.1f} N", (12, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        else:
            cv2.putText(telemetry_panel, "No reachable objects in view", (12, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        # Composite HUD Frame: [ Vision RGB (640x480) | Depth Thumbnail / Telemetry (320x480) ]
        # Resize depth thumbnail to (320x240) and stack above telemetry or place side by side
        depth_small = cv2.resize(depth_color, (320, 240))
        telemetry_small = cv2.resize(telemetry_panel, (320, 240))
        right_column = np.vstack((depth_small, telemetry_small))

        composite = np.hstack((vis_rgb, right_column))
        return composite

    def trigger_gesture(self, gesture_name: str, duration_s: float = 1.0):
        """Allows injecting synthetic gestures from UI/CLI."""
        self.streamer.trigger_synthetic_gesture(gesture_name, duration_s)

    def cycle_target(self):
        """Cycles to next visible object target."""
        if self.visible_targets:
            self.selected_target_idx = (self.selected_target_idx + 1) % len(self.visible_targets)
            self.fusion.selected_target_index = self.selected_target_idx

    def reset_estop(self):
        self.arm.reset_emergency_stop()
        self.state_machine.reset_from_estop()
