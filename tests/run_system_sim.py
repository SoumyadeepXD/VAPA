"""
VAPA End-to-End Automated Simulation Demo
Demonstrates complete multi-modal cycle without user intervention:
Scanning -> Target Selection -> EEG Reach -> Inverse Kinematics -> Trajectory Execution ->
EMG Proportional Force Grasp -> Lifting Object -> EMG Release -> Return Home.
"""

import sys
import os
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vapa_engine import VAPAEngine
from core.state_machine import VAPAState


def run_automated_demo(headless: bool = False):
    print("=" * 75)
    print(" VAPA: Automated Multi-Modal Simulation Demo")
    print("=" * 75)

    engine = VAPAEngine(force_mock=True)
    engine.start()

    window_name = "VAPA Automated Demo [Press Q to Quit]"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 480)

    def _step(duration_s: float):
        t0 = time.time()
        while time.time() - t0 < duration_s:
            frame = engine.get_dashboard_frame()
            if not headless:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    return False
            else:
                time.sleep(0.02)
        return True

    try:
        print("\n[Phase 0] Executing Pre-Flight Diagnostics & Subsystem Self-Test...")
        from tests.test_phase0_preflight import run_phase_0_diagnostics
        if not run_phase_0_diagnostics():
            print("\n[Phase 0] Diagnostics failed. Aborting simulation demo.")
            return

        print("\n[Phase 1] Initializing and scanning 3D room...")
        if not _step(2.5):
            return

        print("\n[Phase 2] Simulating EEG Cognitive Target Selection & Motor Imagery Reach...")
        engine.trigger_gesture("reach", duration_s=1.2)
        if not _step(3.5):
            return

        print("\n[Phase 3] Arm reached target! Simulating EMG Muscle Grasp with Proportional Force...")
        engine.trigger_gesture("grasp", duration_s=1.5)
        if not _step(3.0):
            return

        print("\n[Phase 4] Holding object. Simulating EMG Extensor Release trigger...")
        time.sleep(1.0)
        engine.trigger_gesture("open", duration_s=1.2)
        if not _step(3.5):
            return

        print("\n[Phase 5] Object released. Arm safely returned home.")
        time.sleep(1.0)
        print("\nAutomated demonstration completed successfully!")

    finally:
        engine.stop()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    headless_flag = "--headless" in sys.argv
    run_automated_demo(headless=headless_flag)
