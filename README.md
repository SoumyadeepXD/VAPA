# VAPA — Visually Assisted Prosthetic Arm

[![NVIDIA Jetson](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-modules)
[![RealSense](https://img.shields.io/badge/Camera-Intel%20RealSense%20D435%2FD455-0071C5?logo=intel)](https://www.intelrealsense.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VAPA (Visually Assisted Prosthetic Arm)** is a multi-modal neural-muscular robotic platform running on **NVIDIA Jetson Orin**. It combines **Intel RealSense 3D Depth Perception**, **EEG Brain Wave Decoding**, and **EMG Muscle Activity Sensing** to allow humans to effortlessly control a multi-DOF robotic prosthetic arm to scan the room, recognize objects in 3D metric space, and execute adaptive-force grasps.

---

## 🌟 Key Capabilities

1. **3D Environmental Perception & Spatial Localization**:
   - Intel RealSense RGB-D capture (640x480 @ 30 FPS).
   - Real-time 3D object detection & semantic classification (mugs, bottles, boxes, apples, cans, utensils).
   - Metric 3D Cartesian coordinate deprojection ($X, Y, Z$ in meters) relative to the robotic arm base.
   - Automatic calculation of object physical bounding box dimensions, grasp orientation (pitch/yaw), and approach vectors.

2. **Neural & Muscular Biosignal Decoding (EEG + EMG)**:
   - **EEG Brain Wave Decoding**: Motor Imagery intention decoding (Mu $8-12\text{ Hz}$ & Beta $13-30\text{ Hz}$ Event-Related Desynchronization) to trigger reaches and cycle through detected targets.
   - **EMG Muscle Contraction Decoding**: Surface EMG DSP (50/60Hz notch, 20-450Hz bandpass, moving RMS envelope) for grasp close, hand open, and proportional grip force control ($0.3\text{ N}$ to $10.0\text{ N}$).
   - **Emergency Co-Contraction Safety Abort**: Simultaneous flexor + extensor contraction immediately freezes the arm in place.

3. **Multi-DOF Kinematics & Smooth Motion Planning**:
   - Full 6-DOF / 7-DOF articulated robot arm support (Waist, Shoulder, Elbow, Wrist Pitch, Wrist Roll, Wrist Yaw, Gripper).
   - Closed-form analytical and numerical Damped Least Squares (DLS) Inverse Kinematics solvers (sub-centimeter accuracy).
   - Minimum-jerk quintic polynomial trajectory interpolation preventing abrupt motor vibration.

4. **Multi-Servo Actuation & Closed-Loop Force Control**:
   - **PCA9685 16-Channel 12-bit I2C PWM Driver** on Jetson Orin (`/dev/i2c-1`).
   - **Smart Bus Servos** (Dynamixel / Feetech STS / Hiwonder) via High-Speed Serial UART (`/dev/ttyTHS1`).
   - **Multi-Finger Prosthetic Hand** simultaneous finger drive (Thumb, Index, Middle, Ring, Pinky).
   - Tactile FSR & current feedback closed-loop adaptive grasping.

5. **Plug-and-Play Simulation & Mock Fallback**:
   - Seamless offline testing without any physical hardware attached (generates synthetic RGB-D room scenes and synthetic EMG/EEG waveforms).

---

## 🛠️ Hardware Requirements & Bill of Materials

| Component | Recommended Model | Interface to Jetson |
| :--- | :--- | :--- |
| **Compute Core** | NVIDIA Jetson Orin Nano / Orin NX / AGX Orin | — |
| **3D Camera** | Intel RealSense D435 / D435i / D455 | USB 3.0 Port |
| **Servo Driver** | PCA9685 16-Channel 12-bit I2C Module | Jetson Pin 3 (SDA), Pin 5 (SCL), Pin 1 (3.3V), Pin 6 (GND) |
| **Arm Servos** | 6x High Torque Metal Gear Servos (25kg-60kg·cm) | PCA9685 Channels 0 to 5 |
| **Prosthetic Hand** | Multi-finger hand or 2-finger parallel gripper with FSR | PCA9685 Channels 5 to 10 |
| **Servo Power Supply** | 5V - 7.4V DC (6A - 10A peak) | External Power -> PCA9685 V+ Screw Terminal |
| **Biosignal DAQ** | Arduino / Teensy / ADS1299 / OpenBCI Cyton | USB Serial (`/dev/ttyACM0` @ 115200 baud) |
| **EMG Electrodes** | 2-4 Channel MyoWare / AD8232 / Surface Ag/AgCl | Connected to DAQ Analog Channels A0-A3 |
| **EEG Electrodes** | C3, C4, Cz, Fz (10-20 system) + Ear reference | Connected to DAQ Analog Channels A4-A7 |

---

## 🚀 Installation & Setup on Jetson Orin

### 1. Clone & Navigate to the Repository
```bash
cd ~/Documents/VAPA
```

### 2. Set Up Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Grant User Permissions for I2C and Serial on Jetson
```bash
sudo usermod -a -G i2c,dialout $USER
```
*(Log out and log back in for group permissions to take effect)*

---

## 🎮 Running the System

### A. Run Master System (Interactive HUD Dashboard)
```bash
source .venv/bin/activate

# 1. Run in Simulation / Mock Mode (No physical hardware needed)
python3 vapa_app.py --mock

# 2. Run with Physical RealSense & Jetson Hardware
python3 vapa_app.py --real
```

### B. Interactive HUD Keyboard Shortcuts
| Key | Action |
| :--- | :--- |
| **`[TAB]`** | Cycle through detected 3D objects in the room |
| **`[ R ]`** | Trigger EEG Motor Imagery (Initiate Reach trajectory to locked target) |
| **`[ G ]`** | Trigger EMG Muscle Contraction (Grasp target with adaptive force) |
| **`[ O ]`** | Trigger EMG Muscle Release (Open hand / Release object) |
| **`[ E ]`** | Trigger EMG Co-Contraction **EMERGENCY STOP** (Safety Freeze) |
| **`[ H ]`** | Move Arm to Default **Home Pose** |
| **`[SPACE]`**| Reset from Emergency Stop to Normal Operation |
| **`[ Q ]`** | Safely shut down all threads and exit |

---

## 🧪 Subsystem Verification & Calibration Tools

Run dedicated modular test scripts to verify and calibrate individual subsystems:

### 1. Test 3D Camera & Spatial Perception
```bash
python3 tests/test_realsense.py --mock   # Test synthetic camera
python3 tests/test_realsense.py          # Test physical RealSense
```

### 2. Test Biosignal Processing (EMG & EEG DSP)
```bash
python3 tests/test_biosignals.py
```

### 3. Test Kinematics & Trajectory Interpolation
```bash
python3 tests/test_kinematics.py
```

### 4. Interactive Servo Calibration & Joint Control Tool
```bash
python3 tests/test_servos.py
```

### 5. Test TCA9548A Multiplexer & 4x AS5600 Encoders
```bash
python3 tests/test_tca9548a_encoders.py --mock   # Test simulated encoders
python3 tests/test_tca9548a_encoders.py          # Test physical I2C MUX
```

### 6. Test ESP32 UART Telemetry Stream (/dev/ttyTHS1)
```bash
python3 tests/test_esp32_stream.py --mock        # Test simulated stream
python3 tests/test_esp32_stream.py               # Test physical UART
```

### 7. Automated End-to-End Demonstration Demo
```bash
python3 tests/run_system_sim.py
```

---

## 📂 Repository Structure

```
VAPA/
├── config/
│   ├── system_config.py       # Global dimensions, limits, filter thresholds
│   └── hardware_config.py     # Jetson pinouts, 12-servo map, I2C addresses, UART specs
├── firmware/
│   └── esp32_sensor_node/
│       ├── esp32_sensor_node.ino  # ESP32 C++ firmware (Dual ADS1115 + 100Hz JSON Serial2)
│       └── platformio.ini         # PlatformIO build configuration & library dependencies
├── drivers/
│   ├── tca9548a_as5600.py     # TCA9548A I2C Multiplexer & 4x AS5600 Magnetic Encoder driver
│   └── pca9685_12ch_driver.py # 12-Channel Servo Driver with soft-start S-curve interpolation
├── vision/
│   ├── realsense_camera.py    # RealSense RGB-D capture + Synthetic fallback
│   ├── object_detector.py     # MediaPipe / YOLO / 3D Depth Segmenter
│   ├── spatial_3d.py          # 3D pixel deprojection & grasp pose planning
│   └── visualizer_3d.py       # 3D bounding HUD and depth colormap renderer
├── biosignals/
│   ├── signal_filters.py      # Notch, Butterworth bandpass, RMS envelope, FFT
│   ├── emg_decoder.py         # Muscle activation & proportional force classifier
│   ├── eeg_decoder.py         # Motor imagery (Mu/Beta ERD) brain wave decoder
│   ├── biosignal_streamer.py  # Hardware Serial DAQ & synthetic signal generator
│   ├── esp32_serial_receiver.py # High-speed non-blocking async UART receiver for ESP32
│   └── intent_fusion.py       # Multimodal neural-muscular-vision arbitrator
├── kinematics/
│   ├── arm_model.py           # Multi-DOF arm link dimensions & workspace bounds
│   ├── forward_kinematics.py  # 3D FK & skeleton joint coordinate calculation
│   ├── inverse_kinematics.py  # Analytical geometric & numerical DLS IK solvers
│   └── trajectory_planner.py  # Minimum-jerk smooth joint trajectory generator
├── actuation/
│   ├── servo_interface.py     # Abstract base servo controller
│   ├── pca9685_controller.py  # Jetson Orin 16-channel 12-bit I2C PWM driver
│   ├── serial_servo_controller.py # Smart bus servo serial UART driver
│   ├── mock_arm_controller.py # Software simulation servo controller
│   └── arm_controller.py      # Unified multi-joint arm & multi-finger manager
├── core/
│   ├── state_machine.py       # Master system lifecycle finite state machine
│   └── vapa_engine.py         # Multi-threaded orchestrator uniting all threads
├── tests/
│   ├── test_realsense.py      # Camera & 3D deprojection test
│   ├── test_biosignals.py     # EMG/EEG real-time DSP test
│   ├── test_kinematics.py     # FK/IK reachability test
│   ├── test_servos.py         # Interactive CLI servo calibration tool
│   ├── test_tca9548a_encoders.py # Diagnostic test for TCA9548A + 4x AS5600 encoders
│   ├── test_esp32_stream.py   # Diagnostic test for ESP32 UART telemetry stream
│   └── run_system_sim.py      # End-to-end automated demo
├── vapa_app.py                # Main application with interactive HUD dashboard
├── architecture.md            # Complete circuit schematics, power rails & folder structure
├── context.md                 # Complete system blueprint & architectural context
├── README.md                  # System guide & documentation
└── requirements.txt           # Python dependencies
```

---

## 📖 Architecture & Context Documentation

- For complete circuit schematics, power rail regulation, star grounding topology, and pinouts, see [**`architecture.md`**](architecture.md).
- For mathematical formulations, coordinate transformations, and control state transitions, see [**`context.md`**](context.md).
