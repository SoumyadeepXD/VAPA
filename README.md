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

## 📐 Mathematical Foundations & Algorithmic Principles

VAPA relies on closed-loop mathematical formulations bridging 3D visual perception, biological neural/muscular decoders, kinematics, and smooth actuator dynamics.

```
 [ RGB-D Pixel (u, v) + Z_c ] ──► [ Deprojection (X_c, Y_c, Z_c) ] ──► [ T_base_cam ] ──► Target X_B, Y_B, Z_B
                                                                                               │
 [ Raw EMG / EEG (ADC V) ] ──► [ IIR Notch + Bandpass ] ──► [ RMS / ERD ] ──► Intent & Force  │
                                                                                    │          │
                                                                                    ▼          ▼
                                                   [ Closed-Form & DLS IK ] ◄───────┴──────────┘
                                                                │
                                                                ▼
                                                [ Quintic Minimum-Jerk s(tau) ]
                                                                │
                                                                ▼
                                                [ PCA9685 PWM Ticks & Soft-Move ]
```

### 1. 3D Spatial Perception & Coordinate Transformations

#### A. Pinhole Camera Deprojection
Given pixel coordinates $(u, v)$ on the RGB-D sensor, measured metric depth $Z_C = \text{depth}(u, v)$, focal lengths $(f_x, f_y)$, and optical center $(c_x, c_y)$, the 3D metric coordinates in the **Camera Optical Frame** ($C$) are:

$$X_C = \frac{(u - c_x) \cdot Z_C}{f_x}, \quad Y_C = \frac{(v - c_y) \cdot Z_C}{f_y}, \quad Z_C = \text{depth}(u, v)$$

*Frame Convention*:
* **Camera Optical Frame ($C$)**: $+X_C$ right, $+Y_C$ down, $+Z_C$ forward along optical axis.
* **Robot Arm Base Frame ($B$)**: $+X_B$ forward, $+Y_B$ left, $+Z_B$ upward vertical swivel axis.

#### B. Homogeneous Frame Transformation
The transformation from Camera Optical frame ($C$) to Robot Base frame ($B$) combines intrinsic optical-to-robot axis alignment $\mathbf{R}_{\text{opt}\to\text{rob}}$, physical mounting rotation $\mathbf{R}_{\text{mount}}(\phi, \theta, \psi)$, and translation $\mathbf{t}_B^C$:

$$\mathbf{R}_{\text{opt}\to\text{rob}} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}, \quad \mathbf{R}_{\text{mount}} = \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

$$\mathbf{T}_B^C = \begin{bmatrix} \mathbf{R}_B^C & \mathbf{t}_B^C \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} \mathbf{R}_{\text{mount}} \cdot \mathbf{R}_{\text{opt}\to\text{rob}} & \mathbf{t}_B^C \\ \mathbf{0}^T & 1 \end{bmatrix}$$

$$\begin{bmatrix} X_B \\ Y_B \\ Z_B \\ 1 \end{bmatrix} = \mathbf{T}_B^C \begin{bmatrix} X_C \\ Y_C \\ Z_C \\ 1 \end{bmatrix}$$

#### C. Metric Object Dimensions & Adaptive Grasping
Object physical width $W_{\text{obj}}$ and height $H_{\text{obj}}$ are derived by deprojecting bounding box perimeter coordinates:

$$W_{\text{obj}} = \|\mathbf{P}_{3D}(x_{\max}, y_{\text{mid}}) - \mathbf{P}_{3D}(x_{\min}, y_{\text{mid}})\|_2$$

The required gripper aperture $W_{\text{target}}$ and adaptive gripping force $F_{\text{target}}$ scale dynamically:

$$W_{\text{target}} = \text{clip}(W_{\text{obj}} + \delta_{\text{clearance}}, W_{\min}, W_{\max}) \quad (\delta_{\text{clearance}} = 0.020\text{ m})$$

$$F_{\text{target}} = F_{\min} + (F_{\max} - F_{\min}) \cdot \min\left(1.0, \frac{W_{\text{obj}}}{0.08}\right)$$

---

### 2. Forward Kinematics (FK)

The kinematic chain defines 5 planar link offsets:
* $L_1$: Base pedestal height ($0.100\text{ m}$)
* $L_2$: Upper arm link length ($0.145\text{ m}$)
* $L_3$: Forearm link length ($0.140\text{ m}$)
* $L_4$: Wrist to palm offset ($0.065\text{ m}$)
* $L_5$: Palm to Tool Center Point (TCP) ($0.070\text{ m}$)

Given joint angles $\mathbf{q} = [q_1, q_2, q_3, q_4, q_5]$ (Base Yaw, Shoulder Pitch, Elbow Pitch, Wrist Pitch, Wrist Roll):

1. **Planar Pitch Accumulation**:
   $$\theta_{\text{upper}} = q_2, \quad \theta_{\text{fore}} = q_2 + q_3, \quad \theta_{\text{wrist}} = q_2 + q_3 + q_4$$

2. **Radial and Vertical Projections**:
   $$r_{\text{elbow}} = L_2 \cos(q_2), \quad z_{\text{elbow}} = L_1 + L_2 \sin(q_2)$$
   $$r_{\text{wrist}} = r_{\text{elbow}} + L_3 \cos(\theta_{\text{fore}}), \quad z_{\text{wrist}} = z_{\text{elbow}} + L_3 \sin(\theta_{\text{fore}})$$
   $$r_{\text{palm}} = r_{\text{wrist}} + L_4 \cos(\theta_{\text{wrist}}), \quad z_{\text{palm}} = z_{\text{wrist}} + L_4 \sin(\theta_{\text{wrist}})$$
   $$r_{\text{tcp}} = r_{\text{palm}} + L_5 \cos(\theta_{\text{wrist}}), \quad z_{\text{tcp}} = z_{\text{palm}} + L_5 \sin(\theta_{\text{wrist}})$$

3. **3D Cartesian Tool Center Point**:
   $$\mathbf{p}_{\text{tcp}} = \begin{bmatrix} X_B \\ Y_B \\ Z_B \end{bmatrix} = \begin{bmatrix} r_{\text{tcp}} \cos(q_1) \\ r_{\text{tcp}} \sin(q_1) \\ z_{\text{tcp}} \end{bmatrix}, \quad \text{RPY} = \begin{bmatrix} q_5 \\ \theta_{\text{wrist}} \\ q_1 \end{bmatrix}$$

---

### 3. Inverse Kinematics (IK)

#### A. Analytical Closed-Form Geometric Solution
For a 3D target $[X_B, Y_B, Z_B]$ with approach pitch $\theta_{\text{pitch}}$ and roll $\theta_{\text{roll}}$:

1. **Base Yaw ($q_1$)**:
   $$q_1 = \text{atan2}(Y_B, X_B)$$

2. **Wrist Center Coordinates ($r_w, z_w$)**:
   $$r_{\text{total}} = \sqrt{X_B^2 + Y_B^2}, \quad L_{\text{wrist}} = L_4 + L_5$$
   $$r_w = r_{\text{total}} - L_{\text{wrist}} \cos(\theta_{\text{pitch}}), \quad z_w = Z_B - L_1 - L_{\text{wrist}} \sin(\theta_{\text{pitch}})$$

3. **Reachability Check**:
   $$d = \sqrt{r_w^2 + z_w^2}, \quad |L_2 - L_3| \le d \le (L_2 + L_3)$$

4. **Elbow Pitch ($q_3$) via Law of Cosines**:
   $$\cos(q_3) = \frac{r_w^2 + z_w^2 - L_2^2 - L_3^2}{2 L_2 L_3}$$
   $$q_3 = \text{atan2}\left(-\sqrt{1 - \cos^2(q_3)}, \cos(q_3)\right) \quad (\text{Elbow-up})$$

5. **Shoulder Pitch ($q_2$)**:
   $$\alpha = \text{atan2}(z_w, r_w), \quad \beta = \text{atan2}\left(L_3 \sin(-q_3), L_2 + L_3 \cos(q_3)\right)$$
   $$q_2 = \alpha + \beta$$

6. **Wrist Pitch ($q_4$)**:
   $$q_4 = \theta_{\text{pitch}} - (q_2 + q_3)$$

#### B. Numerical Damped Least Squares (DLS) Jacobian Solver
When nearing kinematic singularities or joint limits, VAPA switches to DLS numerical refinement. For position error $\mathbf{e} = \mathbf{p}_{\text{target}} - f_{\text{FK}}(\mathbf{q})$ and numerical Jacobian $\mathbf{J} \in \mathbb{R}^{3 \times 4}$:

$$\mathbf{J}_{:, j} = \frac{f_{\text{FK}}(\mathbf{q} + \epsilon \hat{\mathbf{e}}_j) - f_{\text{FK}}(\mathbf{q})}{\epsilon}$$

$$\Delta \mathbf{q} = \mathbf{J}^T \left( \mathbf{J} \mathbf{J}^T + \lambda^2 \mathbf{I}_{3 \times 3} \right)^{-1} \mathbf{e}$$

where damping coefficient $\lambda = 0.05$ stabilizes inversion across singular postures.

---

### 4. Minimum-Jerk Smooth Trajectory Interpolation

To eliminate motor gearbox shudder and inertial vibration, trajectories minimize the integral of squared jerk:

$$\min_{\mathbf{q}(t)} \int_0^T \left\| \frac{d^3 \mathbf{q}(t)}{dt^3} \right\|^2 dt$$

subject to boundary constraints $\mathbf{q}(0) = \mathbf{q}_0, \mathbf{q}(T) = \mathbf{q}_1, \dot{\mathbf{q}}(0) = \dot{\mathbf{q}}(T) = \mathbf{0}, \ddot{\mathbf{q}}(0) = \ddot{\mathbf{q}}(T) = \mathbf{0}$.

The solution is a **quintic polynomial** parameterized by normalized time $\tau = \frac{t}{T} \in [0, 1]$:

$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$$

$$\dot{s}(\tau) = \frac{1}{T} \left( 30\tau^2 - 60\tau^3 + 30\tau^4 \right)$$

$$\ddot{s}(\tau) = \frac{1}{T^2} \left( 60\tau - 180\tau^2 + 120\tau^3 \right)$$

Joint position and velocity profiles are evaluated at 50 Hz:

$$\mathbf{q}(t) = \mathbf{q}_{\text{start}} + (\mathbf{q}_{\text{target}} - \mathbf{q}_{\text{start}}) \cdot s(\tau)$$
$$\dot{\mathbf{q}}(t) = (\mathbf{q}_{\text{target}} - \mathbf{q}_{\text{start}}) \cdot \dot{s}(\tau)$$

---

### 5. Biosignal DSP & Decoding Pipeline

#### A. Digital Filter Formulations
* **IIR Notch Filter** (50 Hz / 60 Hz powerline hum rejection, quality factor $Q = 30$):
  $$H_{\text{notch}}(z) = b_0 \frac{1 - 2\cos(\omega_0) z^{-1} + z^{-2}}{1 - 2 r \cos(\omega_0) z^{-1} + r^2 z^{-2}}, \quad \omega_0 = \frac{2\pi f_{\text{notch}}}{f_s}$$

* **4th-Order Butterworth Bandpass**:
  $$|H(j\omega)|^2 = \frac{1}{1 + \left( \frac{\omega^2 - \omega_0^2}{\omega \cdot \text{BW}} \right)^8}$$
  * EMG band: $20\text{ Hz} \le f \le 450\text{ Hz}$
  * EEG band: $1\text{ Hz} \le f \le 45\text{ Hz}$

#### B. Surface EMG Muscle Envelope & Proportional Grip
* **Moving RMS Window** ($W = 50\text{ samples}$):
  $$\text{RMS}[k] = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} x^2[k - i]}$$

* **Normalized Muscle Activation**:
  $$\text{Act} = \text{clip}\left( \frac{\text{RMS} - \text{RMS}_{\text{base}}}{\text{RMS}_{\max} - \text{RMS}_{\text{base}}}, 0.0, 1.0 \right)$$

* **Continuous Proportional Grasp Force**:
  $$F_{\text{grip}} = F_{\min} + (F_{\max} - F_{\min}) \cdot \text{Act}_{\text{flexor}} \quad (F_{\min}=0.3\text{ N}, F_{\max}=10.0\text{ N})$$

* **Co-Contraction Safety Abort**:
  $$\text{Act}_{\text{flexor}} > 0.85 \quad \land \quad \text{Act}_{\text{extensor}} > 0.85 \implies \text{EMERGENCY\_STOP}$$

#### C. EEG Motor Imagery & Attention Trigger
* **FFT Power Spectral Density**:
  $$P(f) = \frac{1}{N} \left| \sum_{n=0}^{N-1} x[n] e^{-j 2\pi f n / f_s} \right|^2$$

* **Event-Related Desynchronization (ERD)** in Mu band ($8-12\text{ Hz}$):
  $$\text{ERD} = \frac{P_{\mu, \text{rest}} - P_{\mu, \text{active}}}{P_{\mu, \text{rest}} + \epsilon}$$
  $$\text{ERD} > 0.35 \implies \text{Intent to Reach Triggered}$$

* **Cognitive Focus / Attention Ratio**:
  $$\text{Attention} = \text{clip}\left( \frac{P_\beta}{P_\mu + \epsilon} \cdot 0.7, 0.0, 1.0 \right) > 0.60 \implies \text{Target Lock Confirmation}$$

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
