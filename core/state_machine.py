"""
VAPA System Finite State Machine
Manages high-level system operating modes, safe transitions, and fault recovery.
"""

import time
import logging

logger = logging.getLogger("VAPA.Core.StateMachine")


class VAPAState:
    """Enumeration of all system operating states."""
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    TARGET_SELECTED = "TARGET_SELECTED"
    PLANNING = "PLANNING"
    REACHING = "REACHING"
    AT_TARGET = "AT_TARGET"
    GRASPING = "GRASPING"
    HOLDING = "HOLDING"
    RETRACTING = "RETRACTING"
    RELEASING = "RELEASING"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class VAPAStateMachine:
    """Controls valid state transitions and tracks state durations."""
    def __init__(self, initial_state=VAPAState.IDLE):
        self._current_state = initial_state
        self._previous_state = None
        self._state_start_time = time.time()
        self._locked_target = None
        logger.info(f"StateMachine initialized in state: {self._current_state}")

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def previous_state(self) -> str:
        return self._previous_state

    @property
    def time_in_state(self) -> float:
        return time.time() - self._state_start_time

    @property
    def locked_target(self):
        return self._locked_target

    def transition_to(self, new_state: str, target=None):
        """Safely transitions to a new state."""
        if new_state == self._current_state and target == self._locked_target:
            return

        # Check for Emergency Stop bypass
        if new_state == VAPAState.EMERGENCY_STOP:
            self._previous_state = self._current_state
            self._current_state = new_state
            self._state_start_time = time.time()
            logger.critical(f"STATE TRANSITION: {self._previous_state} -> EMERGENCY_STOP")
            return

        # Standard transition
        self._previous_state = self._current_state
        self._current_state = new_state
        self._state_start_time = time.time()
        if target is not None:
            self._locked_target = target

        logger.info(f"STATE TRANSITION: {self._previous_state} -> {self._current_state}" + (f" (Target: {target.label})" if target else ""))

    def clear_target(self):
        self._locked_target = None

    def reset_from_estop(self):
        """Recovers from emergency stop into IDLE."""
        if self._current_state == VAPAState.EMERGENCY_STOP:
            self.transition_to(VAPAState.IDLE)
            self.clear_target()
            logger.info("Recovered from EMERGENCY_STOP to IDLE.")
