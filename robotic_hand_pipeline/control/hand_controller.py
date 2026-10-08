
import os
import sys
import time

try:
    from robotic_hand_pipeline import config
except ImportError:
    import config


class HandInterface:
    def __init__(self, port=None):
        self.port = port
        # TODO: open serial/CAN connection here, e.g.:
        # import serial
        # self.conn = serial.Serial(port, baudrate=115200, timeout=0.05)
        self.current_opening_m = 0.10

    def set_opening_width(self, width_m):
        # TODO: convert width_m to your servo's position units and send.
        self.current_opening_m = width_m

    def apply_force_step(self, force_delta_n):
        # TODO: send an incremental force/torque or position command.
        pass

    def read_feedback(self):
        # TODO: read real sensors here (servo current, FSR/tactile, strain gauge).
        return {
            "current_a": None,
            "tactile_n": None,
            "position_m": self.current_opening_m,
        }

    def emergency_stop(self):
        # TODO: cut power / hold position immediately.
        pass


class GraspController:
    def __init__(self, hand: HandInterface):
        self.hand = hand

    def execute_grasp(self, target_width_m, target_force_n, timeout_s=3.0):
        target_force_n = max(config.FORCE_MIN_N, min(config.FORCE_MAX_N, target_force_n))

        self.hand.set_opening_width(target_width_m)
        applied_force = 0.0
        last_current = None

        start = time.time()
        dt = 1.0 / config.CONTROL_LOOP_HZ

        while time.time() - start < timeout_s:
            feedback = self.hand.read_feedback()

            tactile = feedback.get("tactile_n")
            if tactile is not None and tactile >= config.MAX_GRASP_FORCE_HARD_LIMIT_N:
                self.hand.emergency_stop()
                return False

            current = feedback.get("current_a")
            if current is not None and last_current is not None:
                if abs(current - last_current) > config.SLIP_CURRENT_DELTA_THRESHOLD:
                    time.sleep(dt)
                    last_current = current
                    continue
            last_current = current

            if applied_force >= target_force_n:
                break

            step = min(config.CLOSE_STEP_FORCE_N, target_force_n - applied_force)
            self.hand.apply_force_step(step)
            applied_force += step

            time.sleep(dt)

        return applied_force >= target_force_n * 0.9

    def release(self, open_width_m=0.10):
        self.hand.set_opening_width(open_width_m)