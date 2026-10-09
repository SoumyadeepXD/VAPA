# AGENTS.md — VAPA AI Agent System Guide & Operational Contract

> **Single Source of Truth** for any AI coding agent (Antigravity, Claude Code, Codex, Cursor, etc.) working on the **VAPA (Visually Assisted Prosthetic Arm)** repository.
> Read this document completely before performing any code generation, architectural modifications, or hardware driver edits.

---

## 1. Executive Summary & Mission

**VAPA (Visually Assisted Prosthetic Arm)** is a multi-modal robotic prosthetic platform designed for **NVIDIA Jetson Orin** and **ESP32 DevKit V1**. It seamlessly fuses:
1. **3D Environmental Perception & Spatial Localization**: Intel RealSense RGB-D capture (640×480 @ 30 FPS) with 3D pinhole deprojection and spatial object bounding.
2. **Neural & Muscular Biosignal Decoding (EEG + EMG)**: Motor imagery intention decoding (Mu $8-12\text{ Hz}$ & Beta $13-30\text{ Hz}$ ERD) and surface EMG muscle contraction decoding with proportional grip force control ($0.3\text{ N} - 10.0\text{ N}$).
3. **Multi-Joint Kinematics & Motion Planning**: Analytical closed-form geometric and numerical Damped Least Squares (DLS) Inverse Kinematics with quintic minimum-jerk trajectory interpolation.
4. **Multi-Servo Actuation & Closed-Loop Force Control**: 12-channel PCA9685 I2C PWM driver (5x MG996R fingers, 3x DS3225 wrist, 1x DS3218 forearm) and TCA9548A I2C multiplexer with 4x AS5600 12-bit magnetic rotary encoders.
5. **High-Speed Inter-Node Telemetry**: Non-blocking 100 Hz UART JSON streaming between ESP32 and Jetson Orin (`/dev/ttyTHS1` @ 115200 baud).

---

## 2. System Hardware & Electrical Architecture

### 2.1 Power Distribution & Central Star Ground
* **Battery**: 3S LiPo (11.1V nominal, 12.6V peak, 2200–5000mAh, 30C+ discharge).
* **Protection**: HW-287 3S 40A BMS + 30A inline automotive blade fuse + SPST main toggle switch.
* **Power Rails**:
  * **Rail 1 (Direct 11.1V–12.6V)**: Powers NVIDIA Jetson Orin DC barrel jack (5.5mm × 2.5mm center-positive).
  * **Rail 2 (Actuation 6.0V @ 10A–15A Peak)**: High-current step-down buck converter powering PCA9685 $V_+$ screw terminal with a **$4700\,\mu\text{F} / 16\text{V}$ low-ESR decoupling capacitor** across $V_+$ and GND to prevent servo back-EMF brownouts.
  * **Rail 3 (Logic BEC 5.0V @ 3A)**: Regulated 5V step-down module powering ESP32 VIN pin.
* **Star Ground Topology**: A central star ground bus connects Battery Negative (0V), Buck GND, BEC GND, Jetson Pin 6 GND, ESP32 GND, and PCA9685 GND to eliminate ground loops and analog sensor noise.

### 2.2 Jetson Orin 40-Pin Header Pinout
| Pin | Signal | Target Device | Pin Function |
| :--- | :--- | :--- | :--- |
| **Pin 1 / 17** | 3.3V VDD | PCA9685 & TCA9548A | Logic Power (3.3V) |
| **Pin 3** | I2C1_SDA | PCA9685 & TCA9548A | I2C Data (`/dev/i2c-1`) |
| **Pin 5** | I2C1_SCL | PCA9685 & TCA9548A | I2C Clock (`/dev/i2c-1`) |
| **Pin 6 / 9 / 14**| GND | Central Star Ground | System Common Ground |
| **Pin 8** | UART1_TXD | ESP32 GPIO 16 (RX2) | Jetson TX $\to$ ESP32 RX |
| **Pin 10** | UART1_RXD | ESP32 GPIO 17 (TX2) | ESP32 TX $\to$ Jetson RX |

### 2.3 PCA9685 12-Channel Servo Mapping (Section 7B Build Phase)
| Channel | Joint Name | Servo Model | Voltage | Angle Bounds | Home Pose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CH 0** | `finger_thumb` | MG996R | 6.0V | 0° – 180° | 0.0° (Open) |
| **CH 1** | `finger_index` | MG996R | 6.0V | 0° – 180° | 0.0° (Open) |
| **CH 2** | `finger_middle` | MG996R | 6.0V | 0° – 180° | 0.0° (Open) |
| **CH 3** | `finger_ring` | MG996R | 6.0V | 0° – 180° | 0.0° (Open) |
| **CH 4** | `finger_pinky` | MG996R | 6.0V | 0° – 180° | 0.0° (Open) |
| **CH 5** | `joint_wrist_flex` | DS3225 25kg | 6.0V | 0° – 180° | 90.0° (Neutral) |
| **CH 6** | `joint_wrist_rotate` | DS3225 25kg | 6.0V | 0° – 180° | 90.0° (Neutral) |
| **CH 7** | `joint_wrist_bend` | DS3225 25kg | 6.0V | 0° – 180° | 90.0° (Neutral) |
| **CH 8** | `joint_forearm_rotate` | DS3218 20kg | 6.0V | 0° – 180° | 90.0° (Neutral) |
| **CH 9–15**| *Reserved* | — | — | — | Future Expansion |

### 2.4 TCA9548A I2C Multiplexer & 4x AS5600 Encoders
* **Address**: `0x71` on Jetson I2C bus 1 (`/dev/i2c-1`) (A0=VDD to eliminate PCA9685 0x70 ALLCALL collision).
* **Encoder Address**: `0x36` on each multiplexed sub-channel.
  * MUX CH 0: Finger Group Angle (12-bit, 0–4095 ticks, 0.088°/LSB).
  * MUX CH 1: Wrist Flexion Angle.
  * MUX CH 2: Wrist Rotation Angle.
  * MUX CH 3: Forearm Rotation Angle.

### 2.5 ESP32 Node 2 & Analog Sensor Conditioning
* **5x FSR Sensors**: Connected directly to ESP32 internal ADC1 pins (GPIO 32: Thumb, GPIO 33: Index, GPIO 34: Middle, GPIO 35: Ring, GPIO 36: Pinky) with $10\text{ k}\Omega$ pull-down voltage dividers and $100\text{ nF}$ anti-aliasing filter caps ($f_c \approx 159\text{ Hz}$).
* **Dual ADS1115 I2C ADCs (16-bit @ 860 SPS)**:
  * **ADS1115 #1 (`0x48`)**:
    * Channel A0: Flexor EMG (`emg_flex`, MyoWare 2.0 ENV output)
    * Channel A1: Analog EEG brainwave sensor module (`eeg`)
  * **ADS1115 #2 (`0x49`)**:
    * Channel A2: Extensor EMG (`emg_ext`, MyoWare 2.0 ENV output)
* **Hardware Safety & E-Stop Pins**:
  * GPIO 27: Momentary hardware emergency stop button to GND (active-low, internal pullup)
  * GPIO 25: PCA9685 Output Enable (`/OE`) line (active-low enable; driven HIGH to cut servo PWM)
* **Telemetry Streaming (100 Hz Serial2 @ 460800 Baud)**:
  ```json
  {"seq":1425,"fsr":[0.420,0.850,0.120,0.050,0.000],"emg_flex":0.940,"emg_ext":0.120,"emg":0.940,"eeg":0.315,"enc":[125.4],"estop":0,"oe_ok":1,"ts":482910}
  ```

---

## 3. Mathematical Principles & Algorithms

1. **Camera Deprojection**:
   $$X_C = \frac{(u - c_x) Z_C}{f_x}, \quad Y_C = \frac{(v - c_y) Z_C}{f_y}, \quad Z_C = \text{depth}(u, v)$$
2. **Camera-to-Base Coordinate Transform**:
   $$\mathbf{P}_B = \mathbf{T}_B^C \mathbf{P}_C = \begin{bmatrix} \mathbf{R}_{\text{mount}} \cdot \mathbf{R}_{\text{opt}\to\text{rob}} & \mathbf{t}_B^C \\ \mathbf{0}^T & 1 \end{bmatrix} \mathbf{P}_C$$
3. **Analytical Inverse Kinematics (Law of Cosines)**:
   $$r_{\text{total}} = \sqrt{X_B^2 + Y_B^2}, \quad r_w = r_{\text{total}} - L_{\text{wrist}} \cos(\theta_p), \quad z_w = Z_B - L_1 - L_{\text{wrist}} \sin(\theta_p)$$
   $$\cos(q_3) = \frac{r_w^2 + z_w^2 - L_2^2 - L_3^2}{2 L_2 L_3}, \quad q_3 = \text{atan2}(-\sqrt{1 - \cos^2(q_3)}, \cos(q_3))$$
   $$q_1 = \text{atan2}(Y_B, X_B), \quad q_2 = \text{atan2}(z_w, r_w) + \text{atan2}(L_3 \sin(-q_3), L_2 + L_3 \cos(q_3)), \quad q_4 = \theta_p - (q_2 + q_3)$$
4. **Quintic Minimum-Jerk Trajectory Interpolation**:
   $$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5, \quad \tau = \frac{t}{T} \in [0, 1]$$
5. **Biosignal DSP & Safety Invariants**:
   * EMG Moving RMS window: $W = 50\text{ samples}$.
   * Proportional grip force: $F = F_{\min} + (F_{\max} - F_{\min}) \cdot \text{Act}_{\text{flexor}}$.
   * **Emergency Co-Contraction Rule**: Simultaneous Flexor $> 0.85$ and Extensor $> 0.85$ forces immediate `EMERGENCY_STOP` and locks all servos in place.

---

## 4. System Finite State Machine (FSM)

```
 [STARTUP] ──► [PHASE 0: PRE-FLIGHT DIAGNOSTICS & CALIBRATION]
                      │ (All checks passed)
                      ▼
                   [IDLE] ◄──────────────────────────────────────────────┐
                      │ (Objects visible)                                │
                      ▼                                                  │
                 [SCANNING]                                              │
                      │ (EEG Target Cycle / Selection)                   │
                      ▼                                                  │
              [TARGET_SELECTED]                                          │
                      │ (EEG Reach / Motor Imagery Mu ERD)               │
                      ▼                                                  │
                 [PLANNING] ──(IK Feasible)──► [REACHING]                │
                                                   │ (At Target)         │
                                                   ▼                     │
                                              [AT_TARGET]                │
                                                   │ (EMG Grasp Flexor)  │
                                                   ▼                     │
                                              [GRASPING]                 │
                                                   │ (Force Converged)   │
                                                   ▼                     │
                                               [HOLDING]                 │
                                                   │ (EMG Open Extensor) │
                                                   ▼                     │
                                              [RELEASING] ───────────────┘

  [ANY STATE] ──(Co-Contraction / E-Stop Key [E])──► [EMERGENCY_STOP] ──([SPACE])──► [IDLE]
```

---

## 5. Repository Layout

```
VAPA/
├── AGENTS.md                  # This document (Single source of truth for AI agents)
├── README.md                  # Human-facing system overview & quickstart
├── architecture.md            # Hardware wiring, circuit schematics, electrical specs
├── context.md                 # System blueprint & architectural context
├── requirements.txt           # Python dependencies
├── vapa_app.py                # Main application with interactive OpenCV HUD
├── config/
│   ├── system_config.py       # Global dimensions, limits, filter thresholds
│   └── hardware_config.py     # Jetson pinouts, 12-servo map, I2C addresses, UART specs
├── firmware/
│   └── esp32_sensor_node/     # ESP32 C++ firmware (Dual ADS1115 + 100Hz JSON Serial2)
├── drivers/
│   ├── pca9685_12ch_driver.py # 12-Channel Servo Driver (soft-start velocity profiles)
│   └── tca9548a_as5600.py     # TCA9548A I2C MUX + 4x AS5600 Magnetic Encoder driver
├── vision/
│   ├── realsense_camera.py    # RealSense RGB-D capture + Synthetic simulation fallback
│   ├── object_detector.py     # Multi-backend detector (MediaPipe, YOLO, Depth segmenter)
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
│   └── arm_controller.py      # Unified 9-servo prosthetic hand/wrist/forearm controller
├── robotic_hand_pipeline/     # Standalone optical & force grasping experimental pipeline
└── tests/
    ├── test_phase0_preflight.py  # Phase 0: System Pre-Flight Diagnostics & Calibration
    ├── test_phase1_perception.py # Phase 1: 3D Environmental Perception & Spatial Localization
    ├── test_phase2_neural_reach.py # Phase 2: Neural Decoding, Cognitive Selection & Reach Planning
    ├── test_phase3_grasp_force.py # Phase 3: EMG Muscle Grasp & Closed-Loop Force Regulation
    ├── test_phase4_release_retract.py # Phase 4: Object Holding, Extensor Release & Retraction to Home
    ├── test_phase5_e2e_mission.py # Phase 5: Autonomous End-to-End Mission & System Certification
    ├── test_phase6_stress_certification.py # Phase 6: Multi-Cycle Durability, Stress & Fleet Flight Certification
    ├── test_phase7_hand_pipeline.py # Phase 7: Robotic Hand Kinematics, InMoov URDF & Optical Grasping
    ├── test_phase8_flight_qualification.py # Phase 8: Hardware-in-the-Loop Flight Qualification & Telemetry
    ├── test_phase9_production_fleet.py # Phase 9: Full Fleet Production Readiness & Clinical Certification
    ├── test_unit_suite.py        # Automated unit test suite (18 comprehensive tests)
    ├── test_realsense.py         # 3D Camera & object detection verification
    ├── test_biosignals.py        # EMG/EEG real-time DSP verification
    ├── test_kinematics.py        # FK/IK reachability test
    ├── test_servos.py            # 9-Servo interactive CLI calibration tool
    ├── test_tca9548a_encoders.py # TCA9548A I2C MUX & AS5600 encoders test
    ├── test_esp32_stream.py      # ESP32 UART telemetry stream test
    └── run_system_sim.py         # Automated End-to-End Simulation Demo
```

---

## 6. Development & Execution Protocols

### 6.1 Virtual Environment
Always execute Python scripts using the dedicated virtual environment with `PYTHONPATH=.`:
```bash
source .venv/bin/activate
PYTHONPATH=. .venv/bin/python <script_path>
```

### 6.2 Phase 0: Pre-Flight Verification Command
Before running application code or testing physical hardware, execute the Phase 0 pre-flight diagnostic suite:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase0_preflight.py
```
And verify that all unit tests pass:
```bash
PYTHONPATH=. .venv/bin/python tests/test_unit_suite.py
```

### 6.3 Phase 1: 3D Perception & Spatial Localization Verification
To verify RealSense RGB-D capture, semantic object detection, metric deprojection, and grasp target generation:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase1_perception.py --mock
# On physical Jetson Orin with RealSense USB3 camera:
PYTHONPATH=. .venv/bin/python tests/test_phase1_perception.py --real
```

### 6.4 Phase 2: Neural Decoding, Cognitive Selection & Reach Verification
To verify EEG filtering, frontal target cycling, motor imagery Mu ERD, intent fusion, standoff IK, and quintic trajectory execution:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase2_neural_reach.py --mock
# On physical Jetson Orin:
PYTHONPATH=. .venv/bin/python tests/test_phase2_neural_reach.py --real
```

### 6.5 Phase 3: EMG Muscle Grasp & Closed-Loop Force Verification
To verify EMG envelope extraction, flexor activation decoding, proportional force mapping, standoff-to-contact advance, 5-finger PCA9685 flexion, closed-loop tactile force regulation, and over-force safety abort:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase3_grasp_force.py --mock
# On physical Jetson Orin:
PYTHONPATH=. .venv/bin/python tests/test_phase3_grasp_force.py --real
```

### 6.6 Phase 4: Voluntary Release & Home Retraction Verification
To verify object holding stability, extensor muscle release decoding (`HAND_OPEN`), 5-finger synchronized PCA9685 extension, standoff clearance disengagement, quintic retraction trajectory planning, and full FSM cycle back to `IDLE`:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase4_release_retract.py --mock
# On physical Jetson Orin:
PYTHONPATH=. .venv/bin/python tests/test_phase4_release_retract.py --real
```

### 6.7 Phase 5: Autonomous End-to-End Mission & System Certification
To verify full multi-thread loop concurrency, multi-target cycling, autonomous reach and grasp, dynamic co-contraction safety fault injection, emergency stop recovery, and 30+ FPS HUD dashboard telemetry:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase5_e2e_mission.py --mock
# On physical Jetson Orin:
PYTHONPATH=. .venv/bin/python tests/test_phase5_e2e_mission.py --real
```

### 6.8 Phase 6: Multi-Cycle Durability, Real-Time Stress & Fleet Flight Certification
To verify deterministic multi-rate thread latency and timing jitter, 3-cycle manipulation durability, zero joint drift, high-frequency fault transient resilience, sensor blackout tolerance, 100-waypoint PCA9685 boundary pulse invariants, and electrical power rail margins:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase6_stress_certification.py --mock
# On physical Jetson Orin:
PYTHONPATH=. .venv/bin/python tests/test_phase6_stress_certification.py --real
```

### 6.9 Phase 7: Robotic Hand Kinematics, InMoov URDF & Optical Grasping Verification
To verify InMoov URDF kinematic topology, finger joint limits, optical bounding box grasp estimation, FSR tactile calibration curves, multi-finger proportional force regulation, and closed-loop tactile feedback:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase7_hand_pipeline.py --mock
# On physical Jetson Orin:
PYTHONPATH=. .venv/bin/python tests/test_phase7_hand_pipeline.py --real
```

### 6.10 Phase 8: Hardware-in-the-Loop Flight Qualification & Teleoperation Verification
To verify multi-thread concurrency and sub-5ms mutex jitter, keyboard HMI operator controls and re-homing, multi-target spatial disambiguation and standoff IK, full autonomous manipulation cycle (reach, grasp, hold, release, idle), dynamic in-motion co-contraction emergency stop preemption (< 50ms), operator fault recovery, ESP32 UART packet fuzzing tolerance, and 30+ FPS composite HUD telemetry generation:
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase8_flight_qualification.py --mock
# On physical Jetson Orin:
PYTHONPATH=. .venv/bin/python tests/test_phase8_flight_qualification.py --real
```

### 6.11 Phase 9: Full Fleet Production Readiness & Clinical Certification Verification
To verify 3D perception throughput (> 40 FPS), biosignal DSP noise rejection (> 20dB 50Hz notch attenuation, < 1ms DAQ chunk latency), dual-mode kinematic solvers benchmark (Analytical 49/50 vs Numerical DLS 50/50), quintic polynomial boundary velocity and acceleration invariants, 12-channel PCA9685 pulse boundary invariants, 4x AS5600 12-bit magnetic encoder resolution (0.0879°/LSB), complete multimodal conflict arbitration truth table with sub-50ms emergency stop override, 100 Hz UART continuous telemetry with 5-finger tactile physics, and master subsystem health (10/10 operational):
```bash
PYTHONPATH=. .venv/bin/python tests/test_phase9_production_fleet.py --mock
# On physical Jetson Orin:
PYTHONPATH=. .venv/bin/python tests/test_phase9_production_fleet.py --real
```

### 6.12 Running VAPA in Simulation (No Physical Hardware Needed)
```bash
PYTHONPATH=. .venv/bin/python vapa_app.py --mock
# Or automated headless demo:
PYTHONPATH=. .venv/bin/python tests/run_system_sim.py --headless
```

### 6.13 Running VAPA on Physical Jetson Orin Hardware
```bash
sudo usermod -a -G i2c,dialout $USER  # Ensure I2C/UART permissions
PYTHONPATH=. .venv/bin/python vapa_app.py --real
```

---

## 7. Rules & Invariants for AI Coding Agents

1. **Non-Breaking Safety Invariants**:
   * Never bypass or disable the Emergency Stop state transition in `core/state_machine.py` or `actuation/arm_controller.py`.
   * Co-contraction (`Act_flexor > 0.85` and `Act_extensor > 0.85`) must unconditionally preempt all other commands.
2. **Servo Angle Clamping**:
   * All commanded angles must be clipped to their configured hardware limits in `config/hardware_config.py` (strictly $0^\circ \le \theta \le 180^\circ$).
3. **Thread Safety & Non-Blocking Loops**:
   * Actuation, biosignal acquisition, and vision inference run on separate worker threads. Use thread locks when accessing shared joint angles or state buffers.
   * Telemetry receivers must never perform blocking reads that freeze the master HUD rendering loop.
4. **Mock Fallback Parity**:
   * Every physical hardware driver (`pca9685_12ch_driver`, `tca9548a_as5600`, `realsense_camera`, `esp32_serial_receiver`) must preserve mock/synthetic fallback functionality so all code can be verified on machines without physical Jetson/RealSense/ESP32 hardware attached.
5. **Always Run Unit Tests**:
   * After any modification to kinematics, biosignals, drivers, or control logic, run `PYTHONPATH=. .venv/bin/python tests/test_unit_suite.py` to ensure zero regressions.