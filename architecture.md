# VAPA — Complete Hardware & Software Architecture Blueprint

> **System Overview**: VAPA is a distributed, multi-node robotic prosthetic arm platform integrating an **NVIDIA Jetson Orin** (High-Level AI, 3D Vision, Kinematics & Actuation Node) and an **ESP32 Microcontroller** (Bio-Signal & Tactile Acquisition Node) powered by a single high-discharge **3S LiPo battery** with centralized star grounding and multi-rail regulation.

---

## 1. Power Distribution & Grounding Architecture

```
 +-------------------------------------------------------------------------------------------------------+
 |                                  MAIN POWER & BATTERY PROTECTION                                      |
 |                                                                                                       |
 |    [ 3S LiPo Battery ] (11.1V Nominal, 12.6V Max, 2200-5000mAh 30C+)                                  |
 |           │                                                                                           |
 |           ▼                                                                                           |
 |    [ HW-287 3S 40A BMS Board ] (Over-charge, Over-discharge & Short-circuit protection)               |
 |           │                                                                                           |
 |           ▼                                                                                           |
 |    [ 30A Inline Blade Fuse ]                                                                          |
 |           │                                                                                           |
 |           ▼                                                                                           |
 |    [ Main SPST Rocker Switch ]                                                                        |
 |           │                                                                                           |
 +───────────┼───────────────────────────────────┬───────────────────────────────────────────────────────+
             │                                   │                                   │
             ▼                                   ▼                                   ▼
   [ POWER RAIL 1: DIRECT ]            [ POWER RAIL 2: ACTUATION ]          [ POWER RAIL 3: LOGIC BEC ]
     11.1V - 12.6V Switched Bus          11.1V -> 6.0V High-Current           11.1V -> 5.0V Regulated
             │                                   │                                   │
             ▼                                   ▼                                   ▼
    [ DC Barrel Jack ]               [ Servo Buck Converter ]             [ 5V 3A BEC Module ]
    (5.5mm x 2.5mm Center +)         (6.0V DC @ 10A-15A Peak)                        │
             │                                   │                                   │
             ▼                                   ├─► [ 4700µF Low-ESR Cap ]          ▼
    [ NVIDIA JETSON ORIN ]                       │   (Across V+ / GND)        [ ESP32 VIN PIN ]
    (Nano / NX Carrier Board)                    ▼                            (5.0V Input Power)
                                     [ PCA9685 V+ Screw Terminal ]
                                     (High-Current Servo Rail)

 =========================================================================================================
                                     CENTRAL STAR GROUND TOPOLOGY
 =========================================================================================================
                          [ CENTRAL STAR GROUND BUS BAR ]
                                         │
        ┌───────────────┬────────────────┼───────────────┬───────────────┐
        ▼               ▼                ▼               ▼               ▼
   [ BMS Neg ]    [ Servo Buck ]    [ 5V BEC ]     [ Jetson GND ]  [ ESP32 GND ]  [ PCA9685 GND ]
    (Battery 0V)     (GND Out)       (GND Out)       (Pin 6/9/14)     (GND Pin)     (Terminal / Pin)
 =========================================================================================================
```

---

## 2. Node Interconnection & Bus Topology

```
 +-------------------------------------------------------------------------------------------------------+
 |                                  NODE 1: NVIDIA JETSON ORIN                                           |
 |                         (High-Level AI, 3D Vision, Kinematics & Actuation)                            |
 |                                                                                                       |
 |  [ RealSense D435/D455 ] ◄── USB 3.0 (RGB-D 640x480 @ 30 FPS)                                        |
 |                                                                                                       |
 |  [ Jetson I2C Bus 1 (/dev/i2c-1) ]                                                                    |
 |         │                                                                                             |
 |         ├─► [ PCA9685 16-Channel 12-bit PWM Driver @ 0x40 ] (50 Hz Frame Rate)                         |
 |         │         ├── CH 0 : Base Yaw (DS3225 25kg)                                                   |
 |         │         ├── CH 1 : Shoulder Pitch (DS3225 25kg)                                             |
 |         │         ├── CH 2 : Elbow Pitch (DS3225 25kg)                                                |
 |         │         ├── CH 3 : Wrist Flex (DS3218 20kg)                                                 |
 |         │         ├── CH 4 : Wrist Roll (MG90S)                                                       |
 |         │         ├── CH 5 : Wrist Yaw (MG90S)                                                        |
 |         │         └── CH 6-11: 5-Finger Hand Flexion + Thumb Abduction (6x MG90S)                     |
 |         │                                                                                             |
 |         └─► [ TCA9548A 8-Channel I2C Multiplexer @ 0x70 ] (A0/A1/A2 -> GND)                           |
 |                   ├── MUX CH 0 ──► AS5600 Magnetic Encoder #1 (0x36) -> Finger Group Angle            |
 |                   ├── MUX CH 1 ──► AS5600 Magnetic Encoder #2 (0x36) -> Wrist Flex Angle              |
 |                   ├── MUX CH 2 ──► AS5600 Magnetic Encoder #3 (0x36) -> Wrist Rotate Angle            |
 |                   └── MUX CH 3 ──► AS5600 Magnetic Encoder #4 (0x36) -> Forearm Rotate Angle          |
 |                                                                                                       |
 |  [ Hardware UART (/dev/ttyTHS1) ] ◄════════════════════════════════════════════════════════════════╗  |
 +----------------------------------------------------------------------------------------------------║--+
                                              INTER-NODE UART LINK                                    ║
                                      (115200 Baud, 8-N-1, 100 Hz JSON Stream)                        ║
 +----------------------------------------------------------------------------------------------------║--+
 |  [ Serial2 (GPIO 16 RX / GPIO 17 TX) ] ════════════════════════════════════════════════════════════╝  |
 |                                                                                                       |
 |  [ ESP32 Hardware I2C (GPIO 21 SDA / GPIO 22 SCL + 4.7kΩ Pull-ups to 3.3V) ]                          |
 |         │                                                                                             |
 |         ├─► [ ADS1115 ADC #1 @ 0x48 ] (ADDR -> GND, 16-bit, 860 SPS)                                  |
 |         │         ├── CH A0 : FSR 402 Sensor 1 (Thumb Force, 10kΩ divider + 100nF cap)                |
 |         │         ├── CH A1 : FSR 402 Sensor 2 (Index Force, 10kΩ divider + 100nF cap)                |
 |         │         ├── CH A2 : FSR 402 Sensor 3 (Middle Force, 10kΩ divider + 100nF cap)               |
 |         │         └── CH A3 : FSR 402 Sensor 4 (Ring Force, 10kΩ divider + 100nF cap)                 |
 |         │                                                                                             |
 |         └─► [ ADS1115 ADC #2 @ 0x49 ] (ADDR -> 3.3V, 16-bit, 860 SPS)                                 |
 |                   ├── CH A0 : MyoWare 2.0 EMG Sensor (Muscle envelope / raw)                          |
 |                   ├── CH A1 : Analog EEG Sensor Module (Brainwave analog input)                       |
 |                   ├── CH A2 : 10kΩ tied to GND (Unused analog protection)                             |
 |                   └── CH A3 : 10kΩ tied to GND (Unused analog protection)                             |
 |                                                                                                       |
 |                                  NODE 2: ESP32 MICROCONTROLLER                                        |
 |                         (Bio-Signal & Tactile Sensor Acquisition Node)                                |
 +-------------------------------------------------------------------------------------------------------+
```

---

## 3. Circuit Schematics & Pinout Tables

### A. Jetson Orin 40-Pin Header Connections
| Jetson Pin | Signal Name | Target Device | Target Pin | Function / Description |
| :--- | :--- | :--- | :--- | :--- |
| **Pin 1** | 3.3V VDD | PCA9685 & TCA9548A | VCC / VDD | Logic Power (3.3V) |
| **Pin 3** | I2C1_SDA | PCA9685 & TCA9548A | SDA | Main I2C Data (`/dev/i2c-1`) |
| **Pin 5** | I2C1_SCL | PCA9685 & TCA9548A | SCL | Main I2C Clock (`/dev/i2c-1`) |
| **Pin 6** | GND | Central Star Ground | GND | System Common Ground |
| **Pin 8** | UART1_TXD | ESP32 | GPIO 16 (RX2) | Jetson Transmit -> ESP32 Receive |
| **Pin 10** | UART1_RXD | ESP32 | GPIO 17 (TX2) | ESP32 Transmit -> Jetson Receive |

### B. PCA9685 12-Servo Actuator Channel Allocation
| Channel | Joint / Actuator | Servo Model | Voltage | Max Torque | Range | Function |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CH 0** | `joint_1_base_yaw` | **DS3225** | 6.0V | 25 kg·cm | 0° - 180° | Arm Base Waist Swivel |
| **CH 1** | `joint_2_shoulder_pitch`| **DS3225** | 6.0V | 25 kg·cm | 0° - 180° | Shoulder Elevation / Lift |
| **CH 2** | `joint_3_elbow_pitch` | **DS3225** | 6.0V | 25 kg·cm | 0° - 180° | Forearm Flexion / Extension |
| **CH 3** | `joint_4_wrist_pitch` | **DS3218** | 6.0V | 20 kg·cm | 0° - 180° | Wrist Pitch (Flex / Extend) |
| **CH 4** | `joint_5_wrist_roll` | **MG90S** | 6.0V | 2.2 kg·cm | 0° - 180° | Wrist Pronation / Supination |
| **CH 5** | `joint_6_wrist_yaw` | **MG90S** | 6.0V | 2.2 kg·cm | 0° - 180° | Wrist Ulnar / Radial Deviation |
| **CH 6** | `finger_thumb_flex` | **MG90S** | 6.0V | 2.2 kg·cm | 0° - 180° | Thumb Flexion / Close |
| **CH 7** | `finger_index_flex` | **MG90S** | 6.0V | 2.2 kg·cm | 0° - 180° | Index Finger Flexion |
| **CH 8** | `finger_middle_flex`| **MG90S** | 6.0V | 2.2 kg·cm | 0° - 180° | Middle Finger Flexion |
| **CH 9** | `finger_ring_flex` | **MG90S** | 6.0V | 2.2 kg·cm | 0° - 180° | Ring Finger Flexion |
| **CH 10**| `finger_pinky_flex` | **MG90S** | 6.0V | 2.2 kg·cm | 0° - 180° | Little Finger Flexion |
| **CH 11**| `finger_thumb_abduct`| **MG90S** | 6.0V | 2.2 kg·cm | 0° - 180° | Thumb Opposition / Abduction |

### C. TCA9548A Multiplexer to AS5600 12-bit Encoders
| MUX Channel | AS5600 Device | I2C Addr | Joint Tracked | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **SD0 / SC0** | Encoder #1 | `0x36` | **Finger Group Angle** | 12-bit (0-4095 ticks, 0.088°/LSB) |
| **SD1 / SC1** | Encoder #2 | `0x36` | **Wrist Flexion Angle** | 12-bit (0-4095 ticks, 0.088°/LSB) |
| **SD2 / SC2** | Encoder #3 | `0x36` | **Wrist Rotation Angle** | 12-bit (0-4095 ticks, 0.088°/LSB) |
| **SD3 / SC3** | Encoder #4 | `0x36` | **Forearm Rotation Angle** | 12-bit (0-4095 ticks, 0.088°/LSB) |

### D. ESP32 Node 2 Pinouts & Dual ADS1115 ADC Mapping
```
 ESP32 DevKit V1 Pinouts:
 ├── VIN ────────◄ 5.0V from BEC Module
 ├── GND ────────◄ Central Star Ground
 ├── GPIO 21 ────► I2C SDA (with 4.7kΩ pull-up to 3.3V)
 ├── GPIO 22 ────► I2C SCL (with 4.7kΩ pull-up to 3.3V)
 ├── GPIO 16 (RX2) ◄── Jetson Header Pin 8 (UART1_TXD)
 └── GPIO 17 (TX2) ──► Jetson Header Pin 10 (UART1_RXD)

 ADS1115 Module #1 (Address 0x48, ADDR -> GND):
 ├── A0 ◄── FSR 402 Thumb (Voltage Divider: 3.3V -> FSR -> A0 -> 10kΩ Resistor -> GND + 100nF Cap)
 ├── A1 ◄── FSR 402 Index (Voltage Divider: 3.3V -> FSR -> A1 -> 10kΩ Resistor -> GND + 100nF Cap)
 ├── A2 ◄── FSR 402 Middle (Voltage Divider: 3.3V -> FSR -> A2 -> 10kΩ Resistor -> GND + 100nF Cap)
 └── A3 ◄── FSR 402 Ring (Voltage Divider: 3.3V -> FSR -> A3 -> 10kΩ Resistor -> GND + 100nF Cap)

 ADS1115 Module #2 (Address 0x49, ADDR -> 3.3V):
 ├── A0 ◄── MyoWare 2.0 EMG Sensor (Analog Signal Out / ENV)
 ├── A1 ◄── EEG Brainwave Sensor Module (Analog Signal Out)
 ├── A2 ◄── 10kΩ tied to GND (Protection)
 └── A3 ◄── 10kΩ tied to GND (Protection)
```

---

## 4. Inter-Node Protocol & JSON Framing

The ESP32 continuously transmits JSON telemetry frames to the Jetson Orin over **Serial2 (`115200 Baud`) at 100 Hz (every 10ms)**:

```json
{"seq":1425,"fsr":[0.420,0.850,0.120,0.050],"emg":0.940,"eeg":0.315,"ts":482910}
```

### JSON Schema Breakdown:
- **`seq`** (*uint32*): Monotonically increasing packet sequence counter for packet drop detection.
- **`fsr`** (*array of 4 floats*): Filtered voltages ($0.000\text{V} - 3.300\text{V}$) corresponding to Thumb, Index, Middle, Ring fingertip forces.
- **`emg`** (*float*): Filtered MyoWare 2.0 muscle envelope voltage ($0.000\text{V} - 3.300\text{V}$).
- **`eeg`** (*float*): Filtered EEG brainwave analog input voltage.
- **`ts`** (*uint32*): ESP32 internal millisecond timestamp (`millis()`).

---

## 5. Complete Repository Folder Structure

```
VAPA/
├── config/
│   ├── __init__.py                # Package initialization for config
│   ├── system_config.py           # Global kinematic dimensions, limits, filter thresholds
│   └── hardware_config.py         # Jetson Orin pinouts, 12-servo map, I2C addresses, UART specs
├── firmware/
│   └── esp32_sensor_node/
│       ├── esp32_sensor_node.ino  # ESP32 C++/Arduino firmware (Dual ADS1115 + 100Hz JSON Serial2)
│       └── platformio.ini         # PlatformIO build configuration & library dependencies
├── drivers/
│   ├── __init__.py                # Package initialization for hardware drivers
│   ├── tca9548a_as5600.py         # TCA9548A I2C Multiplexer & 4x AS5600 Magnetic Encoder driver
│   └── pca9685_12ch_driver.py     # 12-Channel Servo Driver with soft-start S-curve interpolation
├── vision/
│   ├── __init__.py                # Package initialization for 3D vision
│   ├── realsense_camera.py        # Intel RealSense RGB-D capture + synthetic fallback
│   ├── object_detector.py         # MediaPipe / YOLO / 3D Depth Segmenter
│   ├── spatial_3d.py              # 3D deprojection, Base frame transform, grasp planning
│   └── visualizer_3d.py           # 3D bounding HUD and depth colormap renderer
├── biosignals/
│   ├── __init__.py                # Package initialization for biosignals
│   ├── signal_filters.py          # DSP suite: Notch, Butterworth bandpass, RMS, FFT bandpower
│   ├── emg_decoder.py             # Muscle activation & proportional force classifier
│   ├── eeg_decoder.py             # Motor imagery (Mu/Beta ERD) brain wave decoder
│   ├── biosignal_streamer.py      # Microcontroller serial bridge + synthetic generator
│   ├── esp32_serial_receiver.py   # High-speed non-blocking async UART receiver for ESP32
│   └── intent_fusion.py           # Multimodal neural-muscular-vision arbitrator
├── kinematics/
│   ├── __init__.py                # Package initialization for kinematics
│   ├── arm_model.py               # Arm geometry, link lengths, joint limits, workspace bounds
│   ├── forward_kinematics.py      # 3D FK & intermediate joint skeleton coordinates
│   ├── inverse_kinematics.py      # Analytical geometric & numerical DLS IK solvers
│   └── trajectory_planner.py      # Minimum-jerk quintic polynomial smooth trajectory generator
├── actuation/
│   ├── __init__.py                # Package initialization for actuation
│   ├── servo_interface.py         # Abstract base servo controller interface
│   ├── pca9685_controller.py      # Jetson Orin 16-channel 12-bit I2C PWM servo driver
│   ├── serial_servo_controller.py # Smart bus servo serial UART driver
│   ├── mock_arm_controller.py     # Virtual hardware simulation with simulated current/force
│   └── arm_controller.py          # Unified multi-joint arm & multi-finger hand manager
├── core/
│   ├── __init__.py                # Package initialization for core orchestration
│   ├── state_machine.py           # Master system lifecycle state machine
│   └── vapa_engine.py             # Multi-threaded orchestrator running Vision, Biosignals, Kinematics
├── tests/
│   ├── __init__.py                # Package initialization for tests
│   ├── test_realsense.py          # Camera & 3D deprojection test
│   ├── test_biosignals.py         # Real-time EMG/EEG DSP test
│   ├── test_kinematics.py         # FK/IK reachability & trajectory smoothness test
│   ├── test_servos.py             # Interactive CLI servo calibration tool
│   ├── test_tca9548a_encoders.py  # Diagnostic test for TCA9548A + 4x AS5600 encoders
│   ├── test_esp32_stream.py       # Diagnostic test for ESP32 UART telemetry stream
│   └── run_system_sim.py          # End-to-end automated simulation demo
├── vapa_app.py                    # Master executable application with interactive HUD dashboard
├── architecture.md                # Complete circuit schematics, power rails & folder structure
├── context.md                     # Technical context & mathematical blueprint
├── README.md                      # Comprehensive user guide & capabilities manual
└── requirements.txt               # Unified Python dependencies
```

---

## 6. Software Dependencies & Installation

### A. NVIDIA Jetson Orin (Python 3)
Install Python dependencies into virtual environment:
```bash
pip install smbus2 adafruit-circuitpython-pca9685 pyserial pyrealsense2 opencv-python numpy scipy scikit-learn
```

### B. ESP32 Arduino / PlatformIO Libraries
- **`Adafruit ADS1X15`** (v2.4.2+)
- **`Adafruit BusIO`** (v1.16.1+)
- **`Wire`** (Built-in ESP32 I2C)
