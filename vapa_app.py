"""
================================================================================
VAPA - Visually Assisted Prosthetic Arm
Master Executable Application & Operator HUD Dashboard
================================================================================
Hardware Platform: NVIDIA Jetson Orin Nano + ESP32-WROOM-32
Sensors: Intel RealSense D435i, 5x FSR402 Fingertip Sensors, MyoWare 2.0 EMG, EEG
Actuation: 9x Servos via PCA9685 (5x MG996R, 3x DS3225, 1x DS3218)
"""

import sys
import os
import time
import argparse
import logging
import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.vapa_engine import VAPAEngine
from core.state_machine import VAPAState

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VAPA.App")


def parse_arguments():
    parser = argparse.ArgumentParser(description="VAPA: Visually Assisted Prosthetic Arm Master System")
    parser.add_argument("--mock", action="store_true", help="Force full simulation mode (Mock camera, biosignals & servos)")
    parser.add_argument("--real", action="store_true", help="Force physical hardware mode on Jetson Orin")
    parser.add_argument("--headless", action="store_true", help="Run without graphical OpenCV window (terminal HUD only)")
    return parser.parse_args()


def print_banner():
    banner = r"""
    ======================================================================
     __      __     _      _____            
     \ \    / / /\ | |    |  __ \   /\      
      \ \  / / /  \| |    | |__) | /  \     
       \ \/ / / /\ \ |    |  ___/ / /\ \    
        \  / / ____ \| |__| |    / ____ \   
         \/ /_/    \_\____|_|   /_/    \_\  
     VISUALLY ASSISTED PROSTHETIC ARM SYSTEM
     Jetson Orin Nano + ESP32 + 5x FSRs + 9x Servos (Fingers/Wrist/Forearm)
    ======================================================================
    KEYBOARD CONTROLS (Interactive HUD Window):
      [TAB]   : Cycle through detected 3D vision targets
      [ G ]   : Trigger Grasp (Closes 5 fingers with closed-loop FSR force control)
      [ O ]   : Trigger Open (Opens all 5 fingers fully)
      [ R ]   : Trigger EEG Reach to locked vision target
      [ 1-5 ] : Test individual fingers (1:Thumb, 2:Index, 3:Middle, 4:Ring, 5:Pinky)
      [ W ]   : Test Wrist Flexion (CH5)
      [ F ]   : Test Forearm Rotation (CH8)
      [ H ]   : Move Arm & Hand to Default Home Pose
      [ E ]   : Trigger EMG Co-Contraction EMERGENCY STOP (Safety Freeze)
      [SPACE] : Reset from Emergency Stop to Normal Operation
      [ Q ]   : Gracefully shutdown system
    ======================================================================
    """
    print(banner)


def main():
    args = parse_arguments()
    print_banner()

    force_mock = args.mock
    if not force_mock:
        logger.info("Attempting to connect to physical hardware on Jetson Orin Nano...")
    else:
        logger.info("Running in forced simulation mode (--mock).")

    engine = VAPAEngine(force_mock=force_mock)
    engine.start()

    window_name = "VAPA - Visually Assisted Prosthetic Arm HUD [Press Q to Quit]"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 480)

    logger.info("System fully online. Listening for biosignals and vision targets...")

    finger_states = {"finger_thumb": False, "finger_index": False, "finger_middle": False, "finger_ring": False, "finger_pinky": False}
    wrist_flex_toggle = False
    forearm_rot_toggle = False

    try:
        while True:
            # Generate Unified HUD frame
            dashboard_img = engine.get_dashboard_frame()

            if not args.headless:
                cv2.imshow(window_name, dashboard_img)
                key = cv2.waitKey(20) & 0xFF

                if key == ord("q") or key == 27:  # Q or ESC
                    logger.info("User requested exit.")
                    break
                elif key == 9:  # TAB key
                    engine.cycle_target()
                elif key == ord("g") or key == ord("G"):
                    logger.info("Manual Key: Injected Grasp command.")
                    engine.trigger_gesture("grasp", duration_s=1.5)
                    # If arm is idle/scanning, perform direct manual finger grasp test without blocking UI
                    if engine.state_machine.current_state in (VAPAState.IDLE, VAPAState.SCANNING):
                        logger.info("Manual Key: Direct finger grasp test (IDLE).")
                        engine.arm.close_grasp(close_percent=100.0)
                elif key == ord("o") or key == ord("O"):
                    logger.info("Manual Key: Opening all fingers / Release.")
                    engine.trigger_gesture("open", duration_s=1.5)
                    engine.arm.release_grasp(open_percent=100.0)
                elif key == ord("r") or key == ord("R"):
                    logger.info("Manual Key: Injected EEG Motor Imagery Reach command.")
                    engine.trigger_gesture("reach", duration_s=1.0)
                elif key == ord("1"):
                    finger_states["finger_thumb"] = not finger_states["finger_thumb"]
                    pct = 0.0 if finger_states["finger_thumb"] else 100.0
                    logger.info(f"Test Finger 1 (Thumb): {'Closed' if pct==0.0 else 'Open'}")
                    engine.arm.set_individual_finger("finger_thumb", pct)
                elif key == ord("2"):
                    finger_states["finger_index"] = not finger_states["finger_index"]
                    pct = 0.0 if finger_states["finger_index"] else 100.0
                    logger.info(f"Test Finger 2 (Index): {'Closed' if pct==0.0 else 'Open'}")
                    engine.arm.set_individual_finger("finger_index", pct)
                elif key == ord("3"):
                    finger_states["finger_middle"] = not finger_states["finger_middle"]
                    pct = 0.0 if finger_states["finger_middle"] else 100.0
                    logger.info(f"Test Finger 3 (Middle): {'Closed' if pct==0.0 else 'Open'}")
                    engine.arm.set_individual_finger("finger_middle", pct)
                elif key == ord("4"):
                    finger_states["finger_ring"] = not finger_states["finger_ring"]
                    pct = 0.0 if finger_states["finger_ring"] else 100.0
                    logger.info(f"Test Finger 4 (Ring): {'Closed' if pct==0.0 else 'Open'}")
                    engine.arm.set_individual_finger("finger_ring", pct)
                elif key == ord("5"):
                    finger_states["finger_pinky"] = not finger_states["finger_pinky"]
                    pct = 0.0 if finger_states["finger_pinky"] else 100.0
                    logger.info(f"Test Finger 5 (Pinky): {'Closed' if pct==0.0 else 'Open'}")
                    engine.arm.set_individual_finger("finger_pinky", pct)
                elif key == ord("w") or key == ord("W"):
                    wrist_flex_toggle = not wrist_flex_toggle
                    target_deg = 45.0 if wrist_flex_toggle else 90.0
                    logger.info(f"Test Wrist Flex (CH5): {target_deg} deg")
                    engine.arm.move_to_angles({"joint_wrist_flex": target_deg}, duration_s=0.5)
                elif key == ord("f") or key == ord("F"):
                    forearm_rot_toggle = not forearm_rot_toggle
                    target_deg = 135.0 if forearm_rot_toggle else 90.0
                    logger.info(f"Test Forearm Rotation (CH8): {target_deg} deg")
                    engine.arm.move_to_angles({"joint_forearm_rotate": target_deg}, duration_s=0.5)
                elif key == ord("e") or key == ord("E"):
                    logger.warning("Manual Key: Injected EMERGENCY STOP!")
                    engine.arm.emergency_stop()
                    engine.state_machine.transition_to(VAPAState.EMERGENCY_STOP)
                elif key == ord("h") or key == ord("H"):
                    logger.info("Manual Key: Returning to Home Pose.")
                    engine.arm.go_to_home(duration_s=1.0)
                    engine.state_machine.transition_to(VAPAState.IDLE)
                elif key == 32:  # SPACE bar
                    logger.info("Manual Key: Resetting from Emergency Stop.")
                    engine.reset_estop()
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Shutting down VAPA Engine...")
        engine.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        logger.info("VAPA System safely terminated.")


if __name__ == "__main__":
    main()
