"""
VAPA Unified Arm & Hand Controller
Coordinates multi-joint trajectories, closed-loop force-controlled grasping
using real 5-finger FSR tactile feedback, and 9-servo hardware actuation.

Hardware Safety Gate:
- Checks config/servo_calibration.json upon initialization.
- If missing or unreadable -> REFUSES TO ARM (fails closed).
- Clamps all commanded angles to calibrated per-servo [min_deg, max_deg] bounds.
"""

import os
import json
import time
import logging
import threading
from typing import Optional, Callable, Any
import numpy as np

from config.hardware_config import SIMULATION_MODE, SERVO_CHANNELS, SERVO_ALIASES
from config.system_config import (
    FORCE_MIN_N,
    FORCE_MAX_N,
    FORCE_EMERGENCY_LIMIT_N,
    ACTUATION_LOOP_RATE_HZ,
)
from actuation.servo_interface import BaseServoDriver
from actuation.pca9685_controller import PCA9685ServoDriver
from actuation.serial_servo_controller import SerialServoDriver
from actuation.mock_arm_controller import MockServoDriver
from kinematics.arm_model import ArmModel
from kinematics.trajectory_planner import JointTrajectoryPoint

logger = logging.getLogger("VAPA.Actuation.ArmController")

# Default Home Pose for the 9-Servo Build Phase (Fingers -> Palm -> Wrist -> Forearm)
DEFAULT_BUILD_PHASE_HOME = {
    "finger_thumb": 0.0,          # Fully open (0 deg flexion)
    "finger_index": 0.0,          # Fully open
    "finger_middle": 0.0,         # Fully open
    "finger_ring": 0.0,           # Fully open
    "finger_pinky": 0.0,          # Fully open
    "joint_wrist_flex": 90.0,     # Neutral center
    "joint_wrist_rotate": 90.0,   # Neutral center
    "joint_wrist_bend": 90.0,     # Neutral center
    "joint_forearm_rotate": 90.0, # Neutral center
    "joint_6_gripper": 100.0,     # 100% open
}


class ArmController:
    """Unified controller for arm kinematics, 9-channel PCA9685 servos, and closed-loop FSR grasping."""
    def __init__(
        self,
        driver: BaseServoDriver = None,
        force_mock: bool = False,
        calibration_path: str = "config/servo_calibration.json",
        fsr_calibration_path: str = "config/fsr_calibration.json",
        bench_no_failsafe: bool = False,
        telemetry_provider: Optional[Callable[[], Any]] = None,
    ):
        self.lock = threading.RLock()
        self.arm_model = ArmModel()
        self.dt = 1.0 / ACTUATION_LOOP_RATE_HZ
        self.is_mock = False
        self.calibration_path = calibration_path
        self.fsr_calibration_path = fsr_calibration_path
        self.bench_no_failsafe = bool(bench_no_failsafe)
        self.telemetry_provider = telemetry_provider
        self.servo_calibrated_limits = {}
        self.fsr_calibrated_limits = {}
        self.fsr_force_ceiling_fraction = 0.85
        self.last_bench_warning_time = 0.0
        self.is_armed = False

        # 1. Arming Gate A: Validate Servo Calibration File
        calib_ok = self._load_servo_calibration(calibration_path)

        # 2. Arming Gate B: Validate FSR Tactile Calibration File
        fsr_calib_ok = self._load_fsr_calibration(fsr_calibration_path)

        # 3. Arming Gate C: Validate ESP32 PCA9685 /OE Supervision
        oe_ok = self._check_oe_safety()

        if self.bench_no_failsafe:
            if not oe_ok:
                logger.warning("[BENCH OVERRIDE] OE failsafe check failed, but arming permitted due to --bench-no-failsafe.")
            if not fsr_calib_ok:
                logger.warning("[BENCH OVERRIDE] FSR calibration missing/invalid, but arming permitted due to --bench-no-failsafe.")
            self.is_armed = calib_ok
            self._start_bench_warning_loop()
        else:
            self.is_armed = calib_ok and fsr_calib_ok and oe_ok

        # 4. Driver Discovery
        if driver is not None:
            self.driver = driver
        elif force_mock or SIMULATION_MODE:
            self.driver = MockServoDriver()
            self.is_mock = True
        else:
            # Auto-probe PCA9685 on candidate I2C buses (1, 7, 8, 0)
            pca = PCA9685ServoDriver()
            if pca.is_connected:
                self.driver = pca
                self.is_mock = False
                logger.info("[OK] ArmController connected to physical PCA9685 hardware.")
            else:
                serial_drv = SerialServoDriver()
                if serial_drv.is_connected:
                    self.driver = serial_drv
                    self.is_mock = False
                else:
                    logger.warning("No physical servo hardware detected on I2C/UART. Initializing MockServoDriver.")
                    self.driver = MockServoDriver()
                    self.is_mock = True

        # 4. Initialize arm & hand to Calibrated Home Position
        self.current_angles = DEFAULT_BUILD_PHASE_HOME.copy()
        for j, cfg in self.servo_calibrated_limits.items():
            self.current_angles[j] = cfg["home_deg"]

        for k, v in self.arm_model.home_angles_deg.items():
            if k not in self.current_angles:
                self.current_angles[k] = v

        if self.is_armed:
            self.driver.set_all_angles(self.current_angles)

        self.current_force_n = 0.0
        self.is_emergency_stopped = False
        logger.info(f"ArmController ready (Armed: {self.is_armed}).")

    def _check_oe_safety(self) -> bool:
        """Validates ESP32 PCA9685 /OE supervision hardware configuration and live telemetry."""
        from config.hardware_config import ESP32_PCA9685_OE_ENABLED
        if not ESP32_PCA9685_OE_ENABLED:
            logger.critical(
                "ArmController refusing to arm: ESP32_PCA9685_OE_ENABLED is False in config/hardware_config.py! "
                "Hardware /OE supervision must be enabled. Pass --bench-no-failsafe to override on bench."
            )
            return False

        if self.telemetry_provider is not None:
            try:
                telem = self.telemetry_provider()
                if telem is not None:
                    oe_status = getattr(telem, "oe_ok", False)
                    if not oe_status:
                        logger.critical(
                            "ArmController refusing to arm: ESP32 telemetry reports oe_ok=False! "
                            "Hardware /OE supervision is tripped or inactive. Pass --bench-no-failsafe to override on bench."
                        )
                        return False
            except Exception as e:
                logger.critical(f"Error checking telemetry for oe_ok: {e}. Refusing to arm.")
                return False
        return True

    def _print_bench_warning_if_due(self, force: bool = False):
        now = time.time()
        if self.bench_no_failsafe and (force or (now - self.last_bench_warning_time >= 10.0)):
            self.last_bench_warning_time = now
            msg = (
                "\n" + "*" * 80 + "\n"
                "[LOUD SAFETY WARNING] BENCH OVERRIDE ACTIVE (--bench-no-failsafe)!\n"
                "PCA9685 HARDWARE /OE SUPERVISION IS BYPASSED. SERVOS MAY OPERATE WITHOUT HARDWARE CUTOFF!\n"
                "KEEP PHYSICAL EMERGENCY STOP WITHIN REACH AT ALL TIMES!\n"
                + "*" * 80 + "\n"
            )
            print(msg, flush=True)
            logger.warning(msg)

    def _start_bench_warning_loop(self):
        self._print_bench_warning_if_due(force=True)
        def _warn_worker():
            while getattr(self, "bench_no_failsafe", False):
                time.sleep(10.0)
                self._print_bench_warning_if_due(force=True)
        t = threading.Thread(target=_warn_worker, name="BenchWarningThread", daemon=True)
        t.start()

    def _load_servo_calibration(self, path: str) -> bool:
        """Loads and validates per-servo calibrated min/max/home limits. Fails closed."""
        if not path or not os.path.exists(path):
            logger.critical(f"Servo calibration file '{path}' NOT found! ArmController refusing to arm.")
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if data.get("calibrated") is not True:
                logger.critical(
                    f"Servo calibration file '{path}' is NOT calibrated ('calibrated': false or missing)! "
                    f"Arming refused. Run tools/calibrate_servos.py on physical hardware to calibrate."
                )
                return False
            if not data.get("timestamp"):
                logger.critical(f"Servo calibration file '{path}' is missing timestamp! Arming refused.")
                return False
            if not data.get("tool_version"):
                logger.critical(f"Servo calibration file '{path}' is missing tool_version! Arming refused.")
                return False

            servos = data.get("servos", {})
            if not servos:
                logger.critical(f"Servo calibration file '{path}' has no servos configured! Refusing to arm.")
                return False
            for name, cfg in servos.items():
                min_d = float(cfg.get("min_deg", 0.0))
                max_d = float(cfg.get("max_deg", 180.0))
                home_d = float(cfg.get("home_deg", 90.0))
                self.servo_calibrated_limits[name] = {
                    "min_deg": min_d,
                    "max_deg": max_d,
                    "home_deg": home_d,
                }
            logger.info(f"[ARMED] Calibrated servo bounds loaded ({len(self.servo_calibrated_limits)} servos). ArmController is ARMED.")
            return True
        except Exception as e:
            logger.critical(f"Error parsing servo calibration file '{path}': {e}. Refusing to arm.")
            return False

    def _load_fsr_calibration(self, path: str) -> bool:
        """Loads and validates per-finger calibrated FSR dynamic ranges and fraction-based force ceilings. Fails closed."""
        if not path or not os.path.exists(path):
            logger.critical(f"FSR calibration file '{path}' NOT found! ArmController refusing to arm. Run tools/calibrate_fsr.py.")
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if data.get("calibrated") is not True:
                logger.critical(
                    f"FSR calibration file '{path}' is NOT calibrated ('calibrated': false or missing)! "
                    f"Arming refused. Run tools/calibrate_fsr.py on physical hardware to calibrate."
                )
                return False
            if not data.get("timestamp"):
                logger.critical(f"FSR calibration file '{path}' is missing timestamp! Arming refused.")
                return False
            if not data.get("tool_version"):
                logger.critical(f"FSR calibration file '{path}' is missing tool_version! Arming refused.")
                return False

            sensors = data.get("sensors", {})
            if not sensors:
                logger.critical(f"FSR calibration file '{path}' has no sensors configured! Refusing to arm.")
                return False

            self.fsr_force_ceiling_fraction = float(data.get("force_ceiling_fraction", 0.85))
            for name, cfg in sensors.items():
                tare_v = float(cfg.get("tare_volts", 0.05))
                load_v = float(cfg.get("loaded_volts", 2.80))
                dyn_range = float(cfg.get("dynamic_range_volts", max(0.1, load_v - tare_v)))
                ceil_frac = float(cfg.get("ceiling_fraction", self.fsr_force_ceiling_fraction))
                self.fsr_calibrated_limits[name] = {
                    "channel": int(cfg.get("channel", 0)),
                    "tare_volts": tare_v,
                    "loaded_volts": load_v,
                    "dynamic_range_volts": dyn_range,
                    "ceiling_fraction": ceil_frac,
                    "ceiling_volts": tare_v + ceil_frac * dyn_range,
                }
            logger.info(f"[ARMED] Calibrated FSR parameters loaded ({len(self.fsr_calibrated_limits)} sensors, ceiling={self.fsr_force_ceiling_fraction*100:.1f}%).")
            return True
        except Exception as e:
            logger.critical(f"Error parsing FSR calibration file '{path}': {e}. Refusing to arm.")
            return False

    def clamp_joint_angle(self, joint_name: str, angle: float) -> float:
        """Clamps angle to calibrated hardware limits for servo or kinematic joint bounds."""
        if joint_name in self.servo_calibrated_limits:
            cfg = self.servo_calibrated_limits[joint_name]
            return float(np.clip(angle, cfg["min_deg"], cfg["max_deg"]))
        if joint_name in self.arm_model.limits:
            min_v, max_v, _ = self.arm_model.limits[joint_name]
            return float(np.clip(angle, min_v, max_v))
        if joint_name in SERVO_ALIASES:
            alias = SERVO_ALIASES[joint_name]
            if alias in self.servo_calibrated_limits:
                cfg = self.servo_calibrated_limits[alias]
                return float(np.clip(angle, cfg["min_deg"], cfg["max_deg"]))
            if alias in self.arm_model.limits:
                min_v, max_v, _ = self.arm_model.limits[alias]
                return float(np.clip(angle, min_v, max_v))
        return float(np.clip(angle, 0.0, 180.0))

    def get_joint_angles(self) -> dict[str, float]:
        """Returns current joint angles in degrees."""
        return self.current_angles.copy()

    def go_to_home(self, duration_s: float = 1.5):
        """Moves arm and hand smoothly to default home position."""
        if not self.is_armed:
            logger.warning("ArmController is UNARMED. Ignoring go_to_home.")
            return
        home_pose = DEFAULT_BUILD_PHASE_HOME.copy()
        for j, cfg in self.servo_calibrated_limits.items():
            home_pose[j] = cfg["home_deg"]
        self.move_to_angles(home_pose, duration_s=duration_s)

    def move_to_angles(self, target_angles: dict[str, float], duration_s: float = 1.0):
        """Directly interpolates to target joint configuration."""
        with self.lock:
            if not self.is_armed:
                logger.warning("ArmController is UNARMED. Ignoring move command.")
                return
            if self.is_emergency_stopped:
                logger.warning("Arm is in EMERGENCY STOP state. Ignoring move command.")
                return

            steps = max(5, int(duration_s * ACTUATION_LOOP_RATE_HZ))

            for step in range(1, steps + 1):
                if self.is_emergency_stopped:
                    break
                alpha = step / steps
                interp_angles = {}
                for j, raw_val in target_angles.items():
                    target_val = self.clamp_joint_angle(j, raw_val)
                    q0 = self.current_angles.get(j, target_val)
                    interp_angles[j] = q0 + alpha * (target_val - q0)

                self.driver.set_all_angles(interp_angles)
                self.current_angles.update(interp_angles)
                time.sleep(self.dt)

    def execute_trajectory(self, trajectory: list[JointTrajectoryPoint], abort_check_fn=None) -> bool:
        """
        Executes a planned smooth joint trajectory.
        abort_check_fn: optional callable returning True if execution should abort immediately.
        """
        with self.lock:
            if not self.is_armed:
                logger.warning("ArmController is UNARMED. Refusing trajectory execution.")
                return False
            if self.is_emergency_stopped:
                return False

            logger.info(f"Executing trajectory with {len(trajectory)} waypoints...")
            last_t = 0.0

            for pt in trajectory:
                if self.is_emergency_stopped or (abort_check_fn and abort_check_fn()):
                    logger.warning("Trajectory aborted by safety trigger.")
                    return False

                clamped = self.arm_model.clamp_angles(pt.angles_deg)
                clamped_calib = {j: self.clamp_joint_angle(j, val) for j, val in clamped.items()}
                self.driver.set_all_angles(clamped_calib)
                self.current_angles.update(clamped_calib)

                sleep_time = max(0.001, pt.time_s - last_t)
                time.sleep(sleep_time)
                last_t = pt.time_s

            return True

    def set_gripper_opening_percent(self, percent: float):
        """
        Sets hand opening percentage (0.0 = Fully closed, 100.0 = Fully open).
        Drives all 5 fingers simultaneously (CH 0-4: Thumb, Index, Middle, Ring, Pinky).
        """
        with self.lock:
            if not self.is_armed:
                logger.warning("ArmController is UNARMED. Refusing gripper command.")
                return
            percent = float(np.clip(percent, 0.0, 100.0))
            self.driver.set_joint_angle("joint_6_gripper", percent)
            self.current_angles["joint_6_gripper"] = percent

            # 100% open -> min flexion (open); 0% open -> max flexion (closed)
            for finger in ("finger_thumb", "finger_index", "finger_middle", "finger_ring", "finger_pinky"):
                cfg = self.servo_calibrated_limits.get(finger, {"min_deg": 0.0, "max_deg": 180.0})
                f_min = cfg["min_deg"]
                f_max = cfg["max_deg"]
                flexion_deg = f_min + (1.0 - (percent / 100.0)) * (f_max - f_min)
                flexion_deg = self.clamp_joint_angle(finger, flexion_deg)
                self.driver.set_joint_angle(finger, flexion_deg)
                self.current_angles[finger] = flexion_deg

    def set_individual_finger(self, finger_name: str, percent_open: float):
        """Drives a specific individual finger (0.0 = closed, 100.0 = open)."""
        with self.lock:
            if not self.is_armed:
                logger.warning("ArmController is UNARMED. Refusing finger command.")
                return
            percent = float(np.clip(percent_open, 0.0, 100.0))
            cfg = self.servo_calibrated_limits.get(finger_name, {"min_deg": 0.0, "max_deg": 180.0})
            f_min = cfg["min_deg"]
            f_max = cfg["max_deg"]
            flexion_deg = f_min + (1.0 - (percent / 100.0)) * (f_max - f_min)
            flexion_deg = self.clamp_joint_angle(finger_name, flexion_deg)
            self.driver.set_joint_angle(finger_name, flexion_deg)
            self.current_angles[finger_name] = flexion_deg

    def execute_force_grasp(self, target_force_n: float, timeout_s: float = 3.0, tactile_sensor_fn=None) -> bool:
        """
        Closed-loop force-controlled grasp.
        Closes fingers gradually while reading real FSR feedback from ESP32 until target force is reached.
        """
        with self.lock:
            if not self.is_armed:
                logger.warning("ArmController is UNARMED. Refusing force grasp.")
                return False
            target_force_n = float(np.clip(target_force_n, FORCE_MIN_N, FORCE_MAX_N))
            logger.info(f"Executing force grasp. Target Force: {target_force_n:.2f} N")

            start_time = time.time()
            applied_force = 0.0
            grip_percent = self.current_angles.get("joint_6_gripper", 100.0)
            measured_tactile = None

            while (time.time() - start_time) < timeout_s:
                if self.is_emergency_stopped:
                    logger.warning("Force grasp aborted due to EMERGENCY STOP.")
                    return False

                # 1. Read real tactile feedback from physical FSRs
                measured_tactile = None
                if tactile_sensor_fn is not None:
                    try:
                        measured_tactile = float(tactile_sensor_fn())
                    except Exception:
                        pass

                # Check live telemetry FSR readings against per-finger calibrated dynamic range fraction
                if self.telemetry_provider is not None:
                    try:
                        telem = self.telemetry_provider()
                        if telem is not None and hasattr(telem, "fsr_volts") and telem.fsr_volts:
                            for f_name, f_cfg in self.fsr_calibrated_limits.items():
                                ch = f_cfg["channel"]
                                if ch < len(telem.fsr_volts):
                                    v = telem.fsr_volts[ch]
                                    frac = (v - f_cfg["tare_volts"]) / max(0.05, f_cfg["dynamic_range_volts"])
                                    if frac >= f_cfg["ceiling_fraction"]:
                                        logger.warning(
                                            f"Excessive FSR force on {f_name} ({frac*100:.1f}% >= {f_cfg['ceiling_fraction']*100:.1f}% dynamic range)! Aborting grasp."
                                        )
                                        self.emergency_stop()
                                        return False
                    except Exception:
                        pass

                # Safety check: excessive force fraction or emergency limit
                if measured_tactile is not None:
                    if (
                        measured_tactile >= FORCE_EMERGENCY_LIMIT_N
                        or (0.0 < measured_tactile <= 1.0 and measured_tactile >= self.fsr_force_ceiling_fraction)
                    ):
                        logger.warning(f"Excessive force detected ({measured_tactile:.2f})! Aborting grasp.")
                        self.emergency_stop()
                        return False

                # Check if physical FSR target force reached
                if measured_tactile is not None and measured_tactile >= target_force_n:
                    self.current_force_n = measured_tactile
                    logger.info(f"[SUCCESS] Physical FSR contact force reached: {measured_tactile:.2f}N >= {target_force_n:.2f}N.")
                    return True

                # In mock/simulation mode without live sensors, fall back to simulated force ramp
                if tactile_sensor_fn is None or self.is_mock or (measured_tactile is not None and measured_tactile == 0.0):
                    if applied_force >= target_force_n:
                        self.current_force_n = applied_force
                        logger.info(f"Target force {target_force_n:.2f} N reached.")
                        return True

                # If fingers are fully closed, stop closing further
                if grip_percent <= 0.0:
                    logger.info("Fingers fully closed.")
                    break

                # Step fingers closure by 3%
                grip_percent = max(0.0, grip_percent - 3.0)
                self.set_gripper_opening_percent(grip_percent)

                step_force = 0.25
                applied_force += step_force
                if isinstance(self.driver, MockServoDriver):
                    self.driver.set_tactile_force(applied_force)

                time.sleep(self.dt)

            self.current_force_n = measured_tactile if measured_tactile is not None else applied_force
            return (applied_force >= (target_force_n * 0.8)) or (measured_tactile is not None and measured_tactile >= 0.5)

    def release_grasp(self, open_percent: float = 100.0):
        """Opens fingers fully to release object."""
        with self.lock:
            if not self.is_armed:
                logger.warning("ArmController is UNARMED. Refusing release command.")
                return
            self.set_gripper_opening_percent(open_percent)
            self.current_force_n = 0.0
            if isinstance(self.driver, MockServoDriver):
                self.driver.set_tactile_force(0.0)
            logger.info(f"Grasp released (opened to {open_percent}%).")

    def emergency_stop(self):
        """Immediately halts all motion."""
        with self.lock:
            self.is_emergency_stopped = True
            self.driver.emergency_stop()
            logger.critical("ARM CONTROLLER EMERGENCY STOP TRIGGERED!")

    def reset_emergency_stop(self):
        """Clears emergency stop and re-enables control."""
        with self.lock:
            self.is_emergency_stopped = False
            self.driver.set_all_angles(self.current_angles)
            logger.info("Emergency stop reset.")

    def close(self):
        with self.lock:
            self.driver.close()
