"""
VAPA Hardware Configuration
Defines pinouts, I2C bus numbers, serial ports, PCA9685 servo channel maps,
TCA9548A encoder multiplexer settings, and ESP32 UART interface parameters.
"""

# ==============================================================================
# HARDWARE OPERATION MODE
# ==============================================================================
SIMULATION_MODE = False  # Set to False to run hardware drivers on Jetson Orin

# ==============================================================================
# 1. POWER ARCHITECTURE SPECIFICATIONS
# ==============================================================================
# Main Power: 3S LiPo (11.1V nominal, 12.6V max) via HW-287 3S 40A BMS & 30A Fuse
BATTERY_NOMINAL_VOLTS = 11.1
BATTERY_MAX_VOLTS = 12.6
FUSE_RATING_AMPS = 30.0

# Power Rails:
# - Servo Buck Converter: 11.1V -> 6.0V DC (High Current) to PCA9685 V+ (4700uF cap)
# - Logic BEC: 11.1V -> 5.0V DC to ESP32 VIN
# - Jetson Orin: 11.1V direct via DC Barrel Jack
SERVO_RAIL_VOLTS = 6.0
LOGIC_BEC_VOLTS = 5.0

# ==============================================================================
# 2. JETSON ORIN I2C BUS 1 (/dev/i2c-1)
# ==============================================================================
JETSON_I2C_BUS = 1
PCA9685_I2C_BUS = JETSON_I2C_BUS

# Device 1: PCA9685 16-Channel 12-bit PWM Servo Driver
PCA9685_I2C_ADDRESS = 0x40
PCA9685_PWM_FREQ_HZ = 50  # 50 Hz (20ms standard servo cycle)

# Device 2: TCA9548A 8-Channel I2C Multiplexer (A0=GND, A1=GND, A2=GND)
TCA9548A_I2C_ADDRESS = 0x70

# Sub-Devices: 4x AS5600 12-bit Magnetic Rotary Encoders (Fixed Address 0x36)
AS5600_I2C_ADDRESS = 0x36
ENCODER_MUX_CHANNELS = {
    0: {"name": "finger_group_angle", "description": "Finger Group Flexion Encoder", "zero_offset_deg": 0.0, "direction": 1},
    1: {"name": "wrist_flex_angle",   "description": "Wrist Flexion/Pitch Encoder", "zero_offset_deg": 0.0, "direction": 1},
    2: {"name": "wrist_rotate_angle", "description": "Wrist Pronation/Supination",  "zero_offset_deg": 0.0, "direction": 1},
    3: {"name": "forearm_rotate_angle","description": "Forearm Rotation Encoder",   "zero_offset_deg": 0.0, "direction": 1},
}

# ==============================================================================
# 3. 12-SERVO CHANNEL MAP & CALIBRATION TABLE (PCA9685 @ 0x40)
# ==============================================================================
# Drives 12 total servos: 3x DS3225 (High-Torque), 1x DS3218, 8x MG90S/SG90
SERVO_CHANNELS = {
    # Joint 1: Base Yaw (Waist Swivel) - DS3225 25kg Servo
    "joint_1_base_yaw": {
        "channel": 0, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Joint 2: Shoulder Pitch (Lift) - DS3225 25kg Servo
    "joint_2_shoulder_pitch": {
        "channel": 1, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Joint 3: Elbow Pitch (Flexion) - DS3225 25kg Servo
    "joint_3_elbow_pitch": {
        "channel": 2, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 45.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Joint 4: Wrist Pitch (Flexion/Tilt) - DS3218 20kg Servo
    "joint_4_wrist_pitch": {
        "channel": 3, "type": "DS3218", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Joint 5: Wrist Roll (Pronation/Supination) - MG90S Micro Servo
    "joint_5_wrist_roll": {
        "channel": 4, "type": "MG90S", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Joint 6: Wrist Yaw (Deviation) - MG90S Micro Servo
    "joint_6_wrist_yaw": {
        "channel": 5, "type": "MG90S", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Finger 1: Thumb Flexion - MG90S Micro Servo
    "finger_thumb_flex": {
        "channel": 6, "type": "MG90S", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Finger 2: Index Flexion - MG90S Micro Servo
    "finger_index_flex": {
        "channel": 7, "type": "MG90S", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Finger 3: Middle Flexion - MG90S Micro Servo
    "finger_middle_flex": {
        "channel": 8, "type": "MG90S", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Finger 4: Ring Flexion - MG90S Micro Servo
    "finger_ring_flex": {
        "channel": 9, "type": "MG90S", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Finger 5: Pinky Flexion - MG90S Micro Servo
    "finger_pinky_flex": {
        "channel": 10, "type": "MG90S", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    # Thumb Abduction / Opposition - MG90S Micro Servo
    "finger_thumb_abduct": {
        "channel": 11, "type": "MG90S", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 45.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
}

# ==============================================================================
# 4. INTER-NODE HARDWARE UART (/dev/ttyTHS1 <-> ESP32 Serial2)
# ==============================================================================
JETSON_UART_PORT = "/dev/ttyTHS1"  # Jetson 40-Pin Header Pin 8 (TXD) & Pin 10 (RXD)
JETSON_UART_BAUD = 115200
BIOSIGNAL_SERIAL_PORT = JETSON_UART_PORT
BIOSIGNAL_BAUD_RATE = JETSON_UART_BAUD
BIOSIGNAL_TIMEOUT_S = 0.05
NUM_EMG_CHANNELS = 4
NUM_EEG_CHANNELS = 4
SERIAL_SERVO_PORT = "/dev/ttyTHS1"
SERIAL_SERVO_BAUD_RATE = 1000000

# ESP32 ADC Channel Mapping (Over UART JSON Stream)
# ADS1115 #1 (0x48): FSR Tactile Sensors
FSR_CHANNELS = {
    0: "fsr_thumb",
    1: "fsr_index",
    2: "fsr_middle",
    3: "fsr_ring",
}

# ADS1115 #2 (0x49): Bio-Signals
BIO_CHANNELS = {
    0: "emg_myoware",   # MyoWare 2.0 EMG
    1: "eeg_brainwave", # Analog EEG module
}
