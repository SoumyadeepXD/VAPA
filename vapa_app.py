"""
================================================================================
VAPA - Visually Assisted Prosthetic Arm
Master Executable Application & Operator HUD Dashboard
================================================================================
Integrates Intel RealSense 3D Depth Perception, EEG/EMG Neural Biosignal Decoding,
Inverse Kinematics, and Multi-Servo Robotic Arm Actuation on NVIDIA Jetson Orin.
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
    parser.add_argument("--real", action="store_true", help="Force physical hardware mode (RealSense, I2C PCA9685, Serial DAQ)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical OpenCV window (terminal HUD only)")
    return parser.parse_args()


def print_banner():
    banner = """
    ======================================================================
     __      __     _      _____            
     \ \    / / /\ | |    |  __ \   /\      
      \ \  / / /  \| |    | |__) | /  \     
       \ \/ / / /\ \ |    |  ___/ / /\ \    
        \  / / ____ \| |__| |    / ____ \   
         \/ /_/    \_\____|_|   /_/    \_\  
     VISUALLY ASSISTED PROSTHETIC ARM SYSTEM
     Jetson Orin + RealSense 3D + EEG/EMG Neural Interface + Multi-Servo Arm
    ======================================================================
    KEYBOARD CONTROLS (Interactive HUD Window):
      [TAB]   : Cycle through detected 3D objects in the room
      [ R ]   : Trigger EEG Motor Imagery (Initiate Reach to locked target)
      [ G ]   : Trigger EMG Muscle Contraction (Grasp target with force control)
      [ O ]   : Trigger EMG Muscle Release (Open gripper / Release object)
      [ E ]   : Trigger EMG Co-Contraction EMERGENCY STOP (Safety Freeze)
      [ H ]   : Return Arm to Home Position
      [SPACE] : Reset from Emergency Stop to Normal Operation
      [ Q ]   : Gracefully shutdown system
    ======================================================================
    """
    print(banner)


def main():
    args = parse_arguments()
    print_banner()

    force_mock = args.mock or (not args.real and False)
    logger.info(f"Starting VAPA Engine (force_mock={force_mock}, headless={args.headless})...")

    engine = VAPAEngine(force_mock=force_mock)
    engine.start()

    window_name = "VAPA - Visually Assisted Prosthetic Arm HUD [Press Q to Quit]"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 480)

    logger.info("System fully online. Listening for biosignals and vision targets...")

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
                elif key == ord("r") or key == ord("R"):
                    logger.info("Manual Key: Injected EEG Motor Imagery Reach command.")
                    engine.trigger_gesture("reach", duration_s=1.0)
                elif key == ord("g") or key == ord("G"):
                    logger.info("Manual Key: Injected EMG Muscle Grasp command.")
                    engine.trigger_gesture("grasp", duration_s=1.2)
                elif key == ord("o") or key == ord("O"):
                    logger.info("Manual Key: Injected EMG Muscle Release command.")
                    engine.trigger_gesture("open", duration_s=1.0)
                elif key == ord("e") or key == ord("E"):
                    logger.warning("Manual Key: Injected EMG Co-Contraction EMERGENCY STOP!")
                    engine.trigger_gesture("estop", duration_s=1.5)
                elif key == ord("h") or key == ord("H"):
                    logger.info("Manual Key: Returning arm to Home pose.")
                    engine.arm.go_to_home(duration_s=1.2)
                    engine.state_machine.transition_to(VAPAState.IDLE)
                elif key == 32:  # SPACE bar
                    logger.info("Manual Key: Resetting from Emergency Stop.")
                    engine.reset_estop()
            else:
                # Headless terminal loop
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
