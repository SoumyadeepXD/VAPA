"""
VAPA System Configuration
Defines all global parameters for Vision, Biosignals, Kinematics, Actuation, and Control.
"""

import numpy as np

# ==============================================================================
# 1. VISION & CAMERA CONFIGURATION (Intel RealSense D435 / D435i / D455)
# ==============================================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DEPTH_UNITS_TO_METERS = 0.001  # RealSense standard depth unit is millimeters

# Depth clipping bounds (meters)
MIN_VALID_DEPTH_M = 0.15  # RealSense min range
MAX_VALID_DEPTH_M = 2.50  # Environmental & wall sensing range (manipulation target workspace <= 1.2m)
DEPTH_STAT_METHOD = "median"  # 'median', 'mean', or 'trimmed_mean'

# Object Detection Classes & Confidence Threshold
DETECTION_CONFIDENCE_THRESHOLD = 0.50
MAX_DETECTED_OBJECTS = 6
DETECTED_CLASSES = [
    "mug",
    "cup",
    "bottle",
    "apple",
    "box",
    "can",
    "cell phone",
    "book",
    "spoon",
    "fork",
    "banana",
    "orange",
    "mouse",
    "remote",
]

# Camera Mounting Offset relative to Robot Arm Base (Base Coordinate Frame)
# Base frame: X = Forward (meters), Y = Left (meters), Z = Up (meters)
# Camera frame: X = Right, Y = Down, Z = Forward (Optical RealSense frame)
CAMERA_TO_BASE_TRANSLATION_M = np.array([0.05, 0.15, 0.25], dtype=np.float32)  # [X, Y, Z] in meters
CAMERA_TO_BASE_ROTATION_DEG = np.array([0.0, 15.0, 0.0], dtype=np.float32)     # [Roll, Pitch, Yaw] in degrees

# ==============================================================================
# 2. ROBOTIC ARM GEOMETRY & KINEMATICS (6-DOF + Gripper)
# ==============================================================================
# Arm Link Lengths (in meters)
LINK_LENGTHS = {
    "base_height": 0.080,      # Height from mounting base to shoulder joint axis (L1)
    "upper_arm": 0.180,        # Shoulder to elbow length (L2)
    "forearm": 0.160,          # Elbow to wrist pitch length (L3)
    "wrist_to_palm": 0.085,    # Wrist to palm/gripper center length (L4)
    "finger_length": 0.065,    # Gripper finger reach (L5)
}

# Physical Joint Angle Limits (in degrees)
# [Min, Max, Default Home Position]
JOINT_LIMITS_DEG = {
    "joint_1_base_yaw":       [-90.0, 90.0, 0.0],     # Base swivel (Waist)
    "joint_2_shoulder_pitch": [-30.0, 120.0, 45.0],   # Shoulder lift
    "joint_3_elbow_pitch":    [-150.0, 10.0, -45.0],  # Elbow bend (flexion up to 150 deg)
    "joint_4_wrist_pitch":    [-90.0, 90.0, 0.0],     # Wrist tilt
    "joint_5_wrist_roll":     [-90.0, 90.0, 0.0],     # Wrist rotation / pronation
    "joint_6_gripper":        [0.0, 100.0, 100.0],    # Gripper: 0.0 = Fully closed, 100.0 = Fully open
}

# Gripper Opening Dimensions (in meters)
GRIPPER_MAX_OPENING_M = 0.090  # 90mm fully open
GRIPPER_MIN_OPENING_M = 0.005  # 5mm fully closed

# Maximum Reach and Safety Envelopes (meters)
MAX_ARM_REACH_M = 0.450
MIN_ARM_REACH_M = 0.100
WORKSPACE_BOUNDS_M = {
    "x_min": 0.05, "x_max": 0.45,  # Forward
    "y_min": -0.35, "y_max": 0.35, # Lateral (Left/Right)
    "z_min": -0.05, "z_max": 0.40, # Vertical (Up/Down)
}

# Trajectory Generation Parameters
MAX_JOINT_VELOCITY_DEG_S = 45.0   # Max 45 degrees/sec for smooth, natural movement
MAX_JOINT_ACCELERATION_DEG_S2 = 90.0
TRAJECTORY_INTERPOLATION_HZ = 50  # 50 Hz waypoint generator

# ==============================================================================
# 3. BIOSIGNAL CONFIGURATION (EMG & EEG)
# ==============================================================================
# Sampling Rates
EMG_SAMPLING_RATE_HZ = 1000  # Surface EMG typically sampled at 500-1000 Hz
EEG_SAMPLING_RATE_HZ = 250   # EEG typically sampled at 250 Hz (OpenBCI standard)

# Filter Frequencies (in Hz)
EMG_BANDPASS_LOW_HZ = 20.0
EMG_BANDPASS_HIGH_HZ = 450.0
EMG_NOTCH_HZ = 50.0          # Power line noise notch (50Hz EU/Asia, 60Hz US)
EMG_NOTCH_QUALITY_Q = 30.0

EEG_BANDPASS_LOW_HZ = 1.0
EEG_BANDPASS_HIGH_HZ = 45.0
EEG_NOTCH_HZ = 50.0
EEG_MU_RHYTHM_BAND = (8.0, 12.0)   # Mu rhythm (sensorimotor desynchronization during motor imagery)
EEG_BETA_RHYTHM_BAND = (13.0, 30.0) # Beta rhythm (motor planning & active state)

# EMG Intent Thresholds (Normalized 0.0 to 1.0)
EMG_REST_THRESHOLD = 0.12        # Below this is considered relaxed/rest
EMG_ACTIVATION_THRESHOLD = 0.28  # Muscle contraction detected
EMG_HIGH_CONTRACTION_THRESHOLD = 0.65 # Strong muscle contraction
EMG_CO_CONTRACTION_THRESHOLD = 0.85   # Simultaneous multi-muscle peak -> Emergency Stop / Abort

# EEG Intent Thresholds
EEG_MOTOR_IMAGERY_ERD_THRESHOLD = 0.35  # Relative power drop in Mu band (35% desynchronization = reach trigger)
EEG_ATTENTION_CONFIRM_THRESHOLD = 0.60  # Cognitive focus score threshold for target lock

# Biosignal Buffer Sizes (samples)
EMG_BUFFER_SIZE = 500   # 0.5s window
EEG_BUFFER_SIZE = 500   # 2.0s window
RMS_WINDOW_SIZE = 50    # 50ms moving RMS calculation window

# ==============================================================================
# 4. GRASP FORCE & ADAPTIVE TACTILE CONTROL
# ==============================================================================
# Force limits (Newtons)
FORCE_MIN_N = 0.3
FORCE_MAX_N = 10.0
FORCE_EMERGENCY_LIMIT_N = 12.0

# Class-Specific Target Grasp Force Ranges (Min Force, Max Force in Newtons)
OBJECT_FORCE_MAP_N = {
    "apple": (0.6, 1.8),
    "banana": (0.5, 1.4),
    "orange": (0.8, 2.2),
    "mug": (1.8, 4.5),
    "cup": (1.2, 3.0),
    "bottle": (2.0, 5.5),
    "can": (1.0, 3.0),
    "box": (2.5, 7.0),
    "cell phone": (1.2, 3.2),
    "book": (2.0, 6.0),
    "spoon": (0.8, 2.0),
    "fork": (0.8, 2.0),
    "mouse": (1.0, 2.5),
    "remote": (1.2, 2.8),
    "default": (1.0, 3.5),
}

# ==============================================================================
# 5. CONTROL LOOP TIMING & SYSTEM STATES
# ==============================================================================
MAIN_LOOP_RATE_HZ = 50         # Master orchestrator loop runs at 50 Hz (20ms cycle)
ACTUATION_LOOP_RATE_HZ = 50    # Servo updates at 50 Hz (20ms standard servo PWM cycle)
VISION_PROCESS_RATE_HZ = 15    # Vision detection runs at 15-30 Hz
BIOSIGNAL_PROCESS_RATE_HZ = 100 # Biosignal decoding runs at 100 Hz

# Timeout settings (seconds)
REACH_TIMEOUT_S = 8.0
GRASP_TIMEOUT_S = 4.0
TARGET_LOCK_TIMEOUT_S = 10.0
EMERGENCY_STOP_COOLDOWN_S = 3.0
