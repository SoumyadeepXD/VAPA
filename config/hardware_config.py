"""
VAPA Hardware Configuration
Defines pinouts, I2C bus numbers, serial ports, PCA9685 servo channel maps,
TCA9548A encoder multiplexer settings, and ESP32 UART interface parameters.
Aligned with VAPA Complete Hardware Connection Guide (Fingers -> Palm -> Wrist -> Forearm).
"""

# ==============================================================================
# HARDWARE OPERATION MODE
# ==============================================================================
SIMULATION_MODE = False  # False = Attempt real hardware drivers first on Jetson Orin

# ==============================================================================
# 1. POWER ARCHITECTURE SPECIFICATIONS
# ==============================================================================
# Main Power: 3S LiPo (11.1V nominal, 12.6V max) via BMS 3S 40A (HW-287) & 20A Fuse
BATTERY_NOMINAL_VOLTS = 11.1
BATTERY_MAX_VOLTS = 12.6
FUSE_RATING_AMPS = 20.0

# Power Rails (Hardware Guide Section 1):
# - 6V SERVO RAIL: 300W 20A Buck set to 6.0V -> PCA9685 V+ & all servo RED wires
# - 5V LOGIC RAIL: LM2596 Buck set to 5.0V -> ESP32 VIN, PCA9685 VCC, 5" HDMI display
# - Jetson Power: Official 19V Wall Adapter (19V / 3.42A / 65W) to DC barrel jack
SERVO_RAIL_VOLTS = 6.0
LOGIC_BEC_VOLTS = 5.0

# ==============================================================================
# 2. JETSON ORIN I2C BUS (Pin 3 SDA, Pin 5 SCL)
# ==============================================================================
# Jetson Orin 40-pin header Pin 3 (SDA) and Pin 5 (SCL):
# On JetPack 5/6 Orin Nano, this is typically bus 1, 7, 8, or 0 depending on carrier board.
JETSON_I2C_BUS = 1
JETSON_I2C_CANDIDATE_BUSES = [1, 7, 8, 0]
PCA9685_I2C_BUS = JETSON_I2C_BUS

# Device 1: PCA9685 16-Channel 12-bit PWM Servo Driver (Section 7A: Address 0x40)
PCA9685_I2C_ADDRESS = 0x40
PCA9685_PWM_FREQ_HZ = 50  # 50 Hz (20ms cycle)

# Device 2: TCA9548A 8-Channel I2C Multiplexer (Section 9A: Address 0x70)
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
# 3. 9-SERVO CHANNEL MAP & CALIBRATION TABLE (PCA9685 @ 0x40)
# ==============================================================================
# Matches Section 2.3 & 7B of AGENTS.md and drivers/pca9685_12ch_driver.py:
# CH 0..4: 5x MG996R Finger Servos (Thumb, Index, Middle, Ring, Pinky)
# CH 5..7: 3x DS3225 Wrist Servos (Wrist Flex, Wrist Rotate, Wrist Bend)
# CH 8:    1x DS3218 Forearm Rotation Servo
# CH 9..15: Unconnected / Reserved for future expansion
SERVO_CHANNELS = {
    # --- 5x FINGER SERVOS (MG996R) ---
    "finger_thumb": {
        "channel": 0, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_index": {
        "channel": 1, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_middle": {
        "channel": 2, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_ring": {
        "channel": 3, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_pinky": {
        "channel": 4, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },

    # --- 3x WRIST SERVOS (DS3225 25kg High-Torque) ---
    "joint_wrist_flex": {
        "channel": 5, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "joint_wrist_rotate": {
        "channel": 6, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "joint_wrist_bend": {
        "channel": 7, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },

    # --- 1x FOREARM ROTATION SERVO (DS3218 20kg Servo) ---
    "joint_forearm_rotate": {
        "channel": 8, "type": "DS3218", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },

    # --- Compatibility Aliases for Actuators & Models ---
    "arm_base_yaw": {
        "channel": 5, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "arm_shoulder_pitch": {
        "channel": 6, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "arm_elbow_pitch": {
        "channel": 7, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 45.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "wrist_pitch": {
        "channel": 8, "type": "DS3218", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "joint_1_base_yaw": {
        "channel": 5, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "joint_2_shoulder_pitch": {
        "channel": 6, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "joint_3_elbow_pitch": {
        "channel": 7, "type": "DS3225", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 45.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "joint_4_wrist_pitch": {
        "channel": 8, "type": "DS3218", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 90.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_thumb_flex": {
        "channel": 0, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_index_flex": {
        "channel": 1, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_middle_flex": {
        "channel": 2, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_ring_flex": {
        "channel": 3, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
    "finger_pinky_flex": {
        "channel": 4, "type": "MG996R", "min_us": 500, "max_us": 2500,
        "min_angle_deg": 0.0, "max_angle_deg": 180.0, "home_deg": 0.0,
        "center_offset_deg": 0.0, "direction": 1,
    },
}

# Compatibility aliases mapping kinematic joint names to physical channels
SERVO_ALIASES = {
    "joint_4_wrist_pitch": "joint_wrist_flex",      # Channel 5
    "joint_5_wrist_roll": "joint_wrist_rotate",     # Channel 6
    "joint_6_wrist_yaw": "joint_wrist_bend",        # Channel 7
    "joint_forearm": "joint_forearm_rotate",        # Channel 8
    "finger_thumb_flex": "finger_thumb",            # Channel 0
    "finger_index_flex": "finger_index",            # Channel 1
    "finger_middle_flex": "finger_middle",          # Channel 2
    "finger_ring_flex": "finger_ring",              # Channel 3
    "finger_pinky_flex": "finger_pinky",            # Channel 4
    "finger_little_flex": "finger_pinky",           # Channel 4
    "finger_little": "finger_pinky",                # Channel 4
}

# ==============================================================================
# 4. INTER-NODE HARDWARE UART (Jetson Orin <-> ESP32 Serial2)
# ==============================================================================
# Jetson Pin 8 (UART1_TXD) -> ESP32 GPIO16 (RX2)
# Jetson Pin 10 (UART1_RXD) <- ESP32 GPIO17 (TX2)
# Jetson Pin 9 (GND) <-> ESP32 GND
JETSON_UART_PORT = "/dev/ttyTHS1"
JETSON_UART_CANDIDATES = [
    "/dev/ttyTHS1",  # Primary Jetson 40-pin header UART
    "/dev/ttyTHS0",  # Secondary Jetson UART
    "/dev/ttyUSB0",  # USB-to-UART fallback (if ESP32 plugged via USB cable)
    "/dev/ttyACM0",  # CDC USB fallback
]
JETSON_UART_BAUD = 460800  # High-speed UART baud rate (UART load < 35% with 2-channel EMG)
JETSON_UART_BAUD_CANDIDATES = [460800, 115200]
BIOSIGNAL_SERIAL_PORT = JETSON_UART_PORT
BIOSIGNAL_BAUD_RATE = JETSON_UART_BAUD
BIOSIGNAL_TIMEOUT_S = 0.05
SERIAL_SERVO_PORT = JETSON_UART_PORT
SERIAL_SERVO_BAUD_RATE = 1000000
NUM_EMG_CHANNELS = 2   # Two-site EMG: Flexor on ADS1115 #1 (0x48) A0, Extensor on ADS1115 #2 (0x49) A2
NUM_EEG_CHANNELS = 1   # Single EEG analog OUT on ADS1115 #1 (0x48) A1
NUM_FSR_CHANNELS = 5   # 5x FSR402 sensors on ESP32 GPIO 32-36

# 5x FSR Sensors on ESP32 Analog Pins (GPIO 32, 33, 34, 35, 36)
FSR_CHANNELS = {
    0: "fsr_thumb",   # GPIO 32
    1: "fsr_index",   # GPIO 33
    2: "fsr_middle",  # GPIO 34
    3: "fsr_ring",    # GPIO 35
    4: "fsr_pinky",   # GPIO 36 (alias: fsr_little)
}

FSR_ESP32_PINS = {
    0: 32,  # Thumb (GPIO 32)
    1: 33,  # Index (GPIO 33)
    2: 34,  # Middle (GPIO 34)
    3: 35,  # Ring (GPIO 35)
    4: 36,  # Pinky (GPIO 36)
}

# Dual ADS1115 Addresses & Channel-to-Input Mappings
ADS1115_PRIMARY_ADDR = 0x48    # ADS1115 #1 (Address 0x48)
ADS1115_SECONDARY_ADDR = 0x49  # ADS1115 #2 (Address 0x49)
ADS1115_BIO_ADDR = ADS1115_PRIMARY_ADDR  # Legacy compatibility alias

# Configurable Channel-to-ADS-Input Mapping:
EMG_CHANNEL_MAP = {
    0: {
        "name": "emg_flex",
        "ads_addr": ADS1115_PRIMARY_ADDR,
        "ads_input": 0,  # A0
        "muscle": "flexor_digitorum_superficialis",
        "action": "GRASP_CLOSE",
    },
    1: {
        "name": "emg_ext",
        "ads_addr": ADS1115_SECONDARY_ADDR,
        "ads_input": 2,  # A2
        "muscle": "extensor_digitorum_communis",
        "action": "HAND_OPEN",
    },
}

BIO_CHANNELS = {
    0: "emg_flex",      # ADS1115 #1 A0: Flexor EMG
    1: "eeg_brainwave", # ADS1115 #1 A1: EEG Output
    2: "emg_ext",       # ADS1115 #2 A2: Extensor EMG
}

# ESP32 Hardware Safety Pins
ESP32_PCA9685_OE_PIN = 25       # ESP32 GPIO 25 connected to PCA9685 Output Enable (/OE)
ESP32_PCA9685_OE_ENABLED = False  # Set True when physical /OE wire is attached; defaults off for bench safety
ESP32_ESTOP_BUTTON_PIN = 27     # ESP32 GPIO 27 momentary push button to GND (internal pullup)
ESP32_FORCE_CEILING_N = 12.0    # Firmware hard force limit per FSR tripping OE independent of Jetson
