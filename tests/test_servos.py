"""
VAPA Subsystem Test: 9-Servo Calibration & Manual Joint Controller
Interactive CLI tool to calibrate servo pulse widths (min_us, max_us, trim offset),
test individual joints, and run smooth sinusoidal motion sweeps.
Matches Hardware Connection Guide Section 7B (5x Fingers, 3x Wrist, 1x Forearm).
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actuation.arm_controller import ArmController
from config.hardware_config import SERVO_CHANNELS


def print_menu():
    print("\n" + "=" * 65)
    print(" VAPA 9-SERVO HARDWARE CALIBRATION & JOINT CONTROL MENU")
    print("=" * 65)
    print(" 1. Display Current Joint Angles (CH0 - CH8)")
    print(" 2. Move Specific Joint (e.g. Set Thumb, Wrist Flex, Forearm)")
    print(" 3. Test Hand Open / Close (All 5 Fingers CH0 - CH4)")
    print(" 4. Run Automated Motion Sweep (Fingers & Wrist)")
    print(" 5. Move All Servos to Home Pose")
    print(" 6. Emergency Stop (Kill all PWM)")
    print(" 7. Quit")
    print("=" * 65)


def run_servo_tool():
    force_mock = "--mock" in sys.argv
    print(f"Initializing ArmController on Jetson Orin (force_mock={force_mock})...")
    arm = ArmController(force_mock=force_mock)

    try:
        while True:
            print_menu()
            choice = input("Select an option (1-7): ").strip()

            if choice == "1":
                angles = arm.get_joint_angles()
                print("\nCurrent Joint Angles:")
                for k in (
                    "finger_thumb", "finger_index", "finger_middle", "finger_ring", "finger_pinky",
                    "joint_wrist_flex", "joint_wrist_rotate", "joint_wrist_bend", "joint_forearm_rotate"
                ):
                    val = angles.get(k, 0.0)
                    ch_info = SERVO_CHANNELS.get(k, {})
                    ch_num = ch_info.get("channel", "N/A")
                    s_type = ch_info.get("type", "")
                    print(f"  CH {ch_num:2d} | {k:22s} ({s_type:6s}): {val:5.1f} deg")

            elif choice == "2":
                print("\nAvailable Joints:")
                j_list = [
                    "finger_thumb", "finger_index", "finger_middle", "finger_ring", "finger_pinky",
                    "joint_wrist_flex", "joint_wrist_rotate", "joint_wrist_bend", "joint_forearm_rotate"
                ]
                for idx, j_name in enumerate(j_list):
                    ch = SERVO_CHANNELS[j_name]["channel"]
                    print(f"  {idx + 1}. [CH {ch}] {j_name}")
                j_sel = input("Enter joint number to move (1-9): ").strip()
                if j_sel.isdigit() and 1 <= int(j_sel) <= len(j_list):
                    target_j = j_list[int(j_sel) - 1]
                    cfg = SERVO_CHANNELS[target_j]
                    angle_val = input(f"Enter target angle for {target_j} [{cfg['min_angle_deg']} to {cfg['max_angle_deg']} deg]: ").strip()
                    try:
                        ang_f = float(angle_val)
                        angles = arm.get_joint_angles()
                        angles[target_j] = ang_f
                        print(f"Moving {target_j} to {ang_f} deg...")
                        arm.move_to_angles(angles, duration_s=0.8)
                        print("Done.")
                    except ValueError:
                        print("Invalid angle value.")

            elif choice == "3":
                print("\nTesting Hand Fingers (CH0 to CH4)...")
                print("1. Opening all 5 fingers (100% open)...")
                arm.release_grasp(100.0)
                time.sleep(1.2)
                print("2. Closing all 5 fingers gradually...")
                arm.set_gripper_opening_percent(20.0)
                time.sleep(1.2)
                print("3. Reopening all 5 fingers...")
                arm.release_grasp(100.0)
                print("Hand test complete.")

            elif choice == "4":
                print("\nRunning gentle automated motion sweep on fingers and wrist...")
                for t_step in range(40):
                    t = t_step * 0.15
                    # Sinusoidal sweep fingers open/close and wrist flex
                    f_pct = 50.0 + 40.0 * math.sin(t * 1.5)
                    arm.set_gripper_opening_percent(f_pct)
                    w_flex = 90.0 + 30.0 * math.sin(t * 1.5)
                    arm.driver.set_joint_angle("joint_wrist_flex", w_flex)
                    time.sleep(0.05)
                arm.go_to_home(duration_s=1.0)
                print("Sweep complete. Returned to Home pose.")

            elif choice == "5":
                print("Moving all 9 servos to Home Pose...")
                arm.go_to_home(duration_s=1.0)
                print("Arm at home.")

            elif choice == "6":
                print("Emergency Stop: Killing PWM on all channels!")
                arm.emergency_stop()

            elif choice == "7" or choice.lower() == "q":
                break

    finally:
        arm.close()
        print("\nServo Tool closed.")


if __name__ == "__main__":
    run_servo_tool()
