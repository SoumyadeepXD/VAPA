"""
VAPA Subsystem Test 4: Servo Calibration & Manual Joint Controller
Interactive CLI tool to calibrate servo pulse widths (min_us, max_us, trim offset),
test individual joints, and run smooth sinusoidal motion sweeps.
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actuation.arm_controller import ArmController
from config.hardware_config import SERVO_CHANNELS
from drivers.pca9685_actuator import VAPAActuatorController


def print_menu():
    print("\n" + "=" * 60)
    print(" VAPA SERVO CALIBRATION & JOINT CONTROL MENU")
    print("=" * 60)
    print(" 1. Display Current Joint Angles")
    print(" 2. Move Specific Joint (e.g. set J1 to 30 deg)")
    print(" 3. Test Gripper Open / Close")
    print(" 4. Run Automated Joint Range Sweep (Calibrate Min/Max)")
    print(" 5. Move Arm to Home Pose")
    print(" 6. Direct VAPAActuatorController (CH 0-8) Low-Level Test")
    print(" 7. Emergency Stop (Kill all PWM)")
    print(" 8. Quit")
    print("=" * 60)


def run_servo_tool():
    force_mock = "--mock" in sys.argv or "--real" not in sys.argv
    print(f"Initializing ArmController (force_mock={force_mock})...")
    arm = ArmController(force_mock=force_mock)

    try:
        while True:
            print_menu()
            choice = input("Select an option (1-7): ").strip()

            if choice == "1":
                angles = arm.get_joint_angles()
                print("\nCurrent Joint Angles:")
                for k, v in angles.items():
                    ch_info = SERVO_CHANNELS.get(k, {})
                    ch_num = ch_info.get("channel", "N/A")
                    print(f"  [{k}] (Ch {ch_num}): {v:.1f} deg")

            elif choice == "2":
                print("\nAvailable Joints:")
                j_list = list(SERVO_CHANNELS.keys())
                for idx, j_name in enumerate(j_list):
                    print(f"  {idx + 1}. {j_name}")
                j_sel = input("Enter joint number to move: ").strip()
                if j_sel.isdigit() and 1 <= int(j_sel) <= len(j_list):
                    target_j = j_list[int(j_sel) - 1]
                    cfg = SERVO_CHANNELS[target_j]
                    angle_val = input(f"Enter target angle for {target_j} [{cfg['min_angle_deg']} to {cfg['max_angle_deg']} deg]: ").strip()
                    try:
                        ang_f = float(angle_val)
                        angles = arm.get_joint_angles()
                        angles[target_j] = ang_f
                        print(f"Moving {target_j} to {ang_f} deg...")
                        arm.move_to_angles(angles, duration_s=1.0)
                        print("Done.")
                    except ValueError:
                        print("Invalid angle value.")

            elif choice == "3":
                print("\nTesting Gripper...")
                print("Opening gripper (100%)...")
                arm.release_grasp(100.0)
                time.sleep(1.0)
                print("Closing gripper with force control (5.0 N)...")
                arm.execute_force_grasp(target_force_n=5.0, timeout_s=2.0)
                time.sleep(1.0)
                print("Reopening gripper...")
                arm.release_grasp(100.0)
                print("Gripper test complete.")

            elif choice == "4":
                print("\nRunning automated sinusoidal joint sweep test on all joints...")
                start_angles = arm.get_joint_angles()
                for t_step in range(60):
                    t = t_step * 0.1
                    sweep_angles = start_angles.copy()
                    # Sine sweep base and shoulder gently
                    sweep_angles["joint_1_base_yaw"] = 25.0 * math.sin(t * 1.5)
                    sweep_angles["joint_2_shoulder_pitch"] = 45.0 + 20.0 * math.sin(t * 1.5)
                    sweep_angles["joint_6_gripper"] = 50.0 + 40.0 * math.sin(t * 3.0)
                    arm.driver.set_all_angles(sweep_angles)
                    time.sleep(0.05)
                arm.go_to_home(duration_s=1.0)
                print("Sweep complete. Returned home.")

            elif choice == "5":
                print("Moving arm to Home Pose...")
                arm.go_to_home(duration_s=1.2)
                print("Arm at home.")

            elif choice == "6":
                print("\nDirect Testing VAPAActuatorController (CH 0-8)...")
                try:
                    controller = VAPAActuatorController()
                    print("Testing CH 0 (finger_thumb) -> 90°...")
                    controller.set_servo_angle("finger_thumb", 90)
                    time.sleep(0.5)
                    print("Testing CH 5 (arm_base_yaw) -> 45°...")
                    controller.set_servo_angle("arm_base_yaw", 45)
                    time.sleep(0.5)
                    print("Testing CH 8 (wrist_pitch) -> 30°...")
                    controller.set_servo_angle("wrist_pitch", 30)
                    print("Direct controller test complete.")
                except Exception as e:
                    print(f"Direct actuator test note/error: {e}")

            elif choice == "7":
                print("Emergency Stop Triggered!")
                arm.emergency_stop()

            elif choice == "8" or choice.lower() == "q":
                break

    finally:
        arm.close()
        print("\nServo Tool closed.")


if __name__ == "__main__":
    run_servo_tool()
