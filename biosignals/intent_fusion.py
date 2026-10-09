"""
VAPA Multimodal Intent Fusion Engine
Fuses high-level cognitive EEG intention, dynamic EMG muscle execution,
and 3D visual perception targets into unified robotic arm commands.
"""

import time
import logging
from biosignals.emg_decoder import EMGIntent
from biosignals.eeg_decoder import EEGIntent
from vision.spatial_3d import GraspTarget3D

logger = logging.getLogger("VAPA.Biosignals.Fusion")


class MultimodalCommand:
    """Unified command sent to the robotic arm controller and state machine."""
    # Command types
    NO_OP = "NO_OP"
    CYCLE_TARGET = "CYCLE_TARGET"
    LOCK_TARGET = "LOCK_TARGET"
    START_REACH = "START_REACH"
    START_GRASP = "START_GRASP"
    MODULATE_FORCE = "MODULATE_FORCE"
    RELEASE_GRIP = "RELEASE_GRIP"
    RETURN_HOME = "RETURN_HOME"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    def __init__(
        self,
        action: str,
        target: GraspTarget3D = None,
        target_force_n: float = 2.0,
        gripper_opening_ratio: float = 1.0,
        confidence: float = 1.0,
        source: str = "FUSION",
    ):
        self.action = action
        self.target = target
        self.target_force_n = float(target_force_n)
        self.gripper_opening_ratio = float(gripper_opening_ratio)
        self.confidence = float(confidence)
        self.source = source
        self.timestamp = time.time()

    def __repr__(self):
        target_name = self.target.label if self.target else "None"
        return f"MultimodalCommand(action='{self.action}', target='{target_name}', force={self.target_force_n:.1f}N, src='{self.source}')"


class IntentFusionEngine:
    """
    Multimodal intent arbitrator.
    Prioritizes safety E-stop, then EMG muscle triggers, then EEG cognitive triggers.
    """
    def __init__(self):
        self.selected_target_index = 0
        self.is_target_locked = False
        self.current_state = "IDLE"

    def fuse(
        self,
        emg_intent: EMGIntent,
        eeg_intent: EEGIntent,
        visible_targets: list[GraspTarget3D],
        system_state: str,
        estop_hardware_trigger: bool = False,
    ) -> MultimodalCommand:
        """
        Fuses biosignals and vision into a unified action command.
        """
        self.current_state = system_state
        num_targets = len(visible_targets)

        # 1. HIGHEST PRIORITY: Emergency Stop Trigger (Hardware E-Stop Button or EMG Co-contraction)
        if estop_hardware_trigger:
            logger.critical("ESP32 Hardware E-Stop button pressed! Triggering EMERGENCY_STOP.")
            return MultimodalCommand(
                action=MultimodalCommand.EMERGENCY_STOP,
                confidence=1.0,
                source="ESP32_BUTTON_ESTOP",
            )

        if emg_intent.gesture == EMGIntent.CO_CONTRACTION_ESTOP:
            logger.warning("EMG Co-contraction detected! Triggering EMERGENCY_STOP.")
            return MultimodalCommand(
                action=MultimodalCommand.EMERGENCY_STOP,
                confidence=emg_intent.confidence,
                source="EMG_ESTOP",
            )

        # Ensure selected target index is within bounds
        if num_targets > 0:
            self.selected_target_index = self.selected_target_index % num_targets
            active_target = visible_targets[self.selected_target_index]
        else:
            active_target = None

        # 2. STATE-BASED MULTIMODAL ARBITRATION:

        # --- A. SCANNING / IDLE STATE ---
        if system_state in ("IDLE", "SCANNING"):
            # EEG Trigger to cycle between detected 3D objects
            if eeg_intent.command == EEGIntent.TARGET_CYCLE_NEXT and num_targets > 1:
                self.selected_target_index = (self.selected_target_index + 1) % num_targets
                logger.info(f"Cycled target to #{self.selected_target_index+1}: {visible_targets[self.selected_target_index].label}")
                return MultimodalCommand(
                    action=MultimodalCommand.CYCLE_TARGET,
                    target=visible_targets[self.selected_target_index],
                    source="EEG_CYCLE",
                )

            # EEG Motor Imagery or Attention -> Lock Target & Start Reach
            if active_target and (
                eeg_intent.command in (EEGIntent.TARGET_LOCK_CONFIRM, EEGIntent.INTENT_REACH)
                or emg_intent.activation_level > 0.40
            ):
                self.is_target_locked = True
                return MultimodalCommand(
                    action=MultimodalCommand.START_REACH,
                    target=active_target,
                    target_force_n=active_target.target_force_n,
                    confidence=max(eeg_intent.confidence, emg_intent.confidence),
                    source="EEG_MOTOR_IMAGERY" if eeg_intent.motor_imagery_active else "EMG_INTENT",
                )

        # --- B. REACHING STATE ---
        elif system_state == "REACHING":
            # EMG Release / Open intent during reach aborts and returns home
            if emg_intent.gesture == EMGIntent.HAND_OPEN and emg_intent.confidence > 0.7:
                return MultimodalCommand(
                    action=MultimodalCommand.RETURN_HOME,
                    source="EMG_ABORT",
                )

        # --- C. AT TARGET / GRASPING STATE ---
        elif system_state in ("AT_TARGET", "GRASPING"):
            # EMG Grasp Close intent initiates grip closure with proportional force
            if emg_intent.gesture in (EMGIntent.GRASP_CLOSE, EMGIntent.PINCH):
                target_force = active_target.target_force_n if active_target else emg_intent.proportional_force_n
                # Proportional force modulated by muscle activation
                effective_force = max(target_force * 0.5, emg_intent.proportional_force_n)
                return MultimodalCommand(
                    action=MultimodalCommand.START_GRASP,
                    target=active_target,
                    target_force_n=effective_force,
                    confidence=emg_intent.confidence,
                    source="EMG_GRASP",
                )

        # --- D. HOLDING STATE ---
        elif system_state == "HOLDING":
            # Continuous EMG flexor activity modulates grip firmness
            if emg_intent.gesture == EMGIntent.GRASP_CLOSE:
                return MultimodalCommand(
                    action=MultimodalCommand.MODULATE_FORCE,
                    target_force_n=emg_intent.proportional_force_n,
                    source="EMG_PROPORTIONAL_FORCE",
                )
            # EMG Hand Open intent triggers object release
            elif emg_intent.gesture == EMGIntent.HAND_OPEN:
                return MultimodalCommand(
                    action=MultimodalCommand.RELEASE_GRIP,
                    source="EMG_RELEASE",
                )

        return MultimodalCommand(action=MultimodalCommand.NO_OP)
