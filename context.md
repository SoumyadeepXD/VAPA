# VAPA: Visually Assisted Prosthetic Arm — System Context & Architecture Blueprint

> **System Overview**: VAPA is an advanced multi-modal robotic & prosthetic arm platform that fuses **3D Computer Vision** (Intel RealSense RGB-D on NVIDIA Jetson Orin), **Neural & Muscular Biosignal Decoding** (EEG Brain Waves & EMG Muscle Activity), and **Multi-Joint Kinematics & Actuation** (Multi-Servo Robotic Arm + Adaptive Force Gripper).

---

## 1. System Architecture Diagram

```
 +-----------------------------------------------------------------------------------+
 |                             NVIDIA JETSON ORIN                                    |
 |                                                                                   |
 |  [ 3D Perception Pipeline ]                                                        |
 |  Intel RealSense D435/D455                                                        |
 |         │                                                                         |
 |         ▼                                                                         |
 |   RGB-D Frames (640x480 @ 30Hz)                                                   |
 |         │                                                                         |
 |         ├─► Object Detection (MediaPipe / DepthGeometric Segmenter)                |
 |         │         │                                                               |
 |         │         ▼                                                               |
 |         └─► 3D Spatial Localization (Deprojection + T_base_cam Transform)         |
 |                   │                                                               |
 |                   ▼                                                               |
 |             GraspTarget3D [X, Y, Z, Orientation, Force, Opening]                  |
 |                   │                                                               |
 |                   ▼                                                               |
 |  [ Multimodal Intent Fusion Engine ] ◄─── State Machine (IDLE -> SCAN -> REACH..) |
 |         ▲                        ▲                                                |
 |         │                        │                                                |
 |  [ EMG Muscle Decoder ]   [ EEG Brain Decoder ]                                   |
 |  (Flexor / Extensor)      (Motor Imagery C3/Cz/Fz)                                |
 |  - Grasp Close Trigger    - Object Selection Trigger                              |
 |  - Proportional Force     - Reach Intention (Mu ERD)                              |
 |  - Co-contraction E-Stop  - Cognitive Confirmation                                |
 |         ▲                        ▲                                                |
 |         └────────┬───────────────┘                                                |
 |                  │                                                                |
 |          Serial DAQ / USB Bridge (Arduino / Teensy / ADS1299)                     |
 |                                                                                   |
 |  [ Kinematics & Motion Planning ]                                                 |
 |         │                                                                         |
 |         ├─► Analytical / Numerical Damped Least Squares (DLS) IK Solver           |
 |         ├─► Minimum-Jerk Smooth Trajectory Planner (Quintic Polynomial)          |
 |         │                                                                         |
 |         ▼                                                                         |
 |  [ Multi-Servo Actuation & Closed-Loop Force Control ]                            |
 |         │                                                                         |
 |         ├─► PCA9685 16-Channel 12-bit I2C PWM Driver (/dev/i2c-1 @ 50Hz)          |
 |         ├─► Serial Bus Servos (STS3215 / Dynamixel / Hiwonder @ 1Mbps)            |
 |         └─► Multi-Finger Hand (Thumb, Index, Middle, Ring, Pinky) + FSR Tactile   |
 +-----------------------------------------------------------------------------------+
```

---

## 2. Hardware Wiring & Pinout Mapping

### A. Jetson Orin 40-Pin Header to PCA9685 16-Channel PWM Servo Driver
| Jetson Orin Pin | Pin Name | PCA9685 Pin | Description |
| :--- | :--- | :--- | :--- |
| **Pin 1 / Pin 17** | 3.3V VDD | **VCC** | Logic Power (3.3V) |
| **Pin 3** | I2C1_SDA | **SDA** | I2C Data Line (`/dev/i2c-1`) |
| **Pin 5** | I2C1_SCL | **SCL** | I2C Clock Line (`/dev/i2c-1`) |
| **Pin 6 / Pin 9** | GND | **GND** | Common System Ground |
| **External 5V-7.4V (5A-10A)** | Battery / Power Supply | **V+ (Screw Terminal)** | Servo Motor High-Current Power (Do **NOT** power servos from Jetson 5V pins) |

### B. PCA9685 Servo Channel Assignment
| Channel | Joint Name | Servo Type / Torque | Motion Range | Default Home |
| :--- | :--- | :--- | :--- | :--- |
| **CH 0** | `joint_1_base_yaw` | High Torque (25-35 kg·cm) | -90° to +90° (180°) | 0.0° (Facing forward) |
| **CH 1** | `joint_2_shoulder_pitch` | High Torque (35-60 kg·cm) | -30° to +120° (150°) | +45.0° (Elevated) |
| **CH 2** | `joint_3_elbow_pitch` | High Torque (25-35 kg·cm) | -150° to +10° (160°) | -45.0° (Bent) |
| **CH 3** | `joint_4_wrist_pitch` | Standard (15-20 kg·cm) | -90° to +90° (180°) | 0.0° (Level) |
| **CH 4** | `joint_5_wrist_roll` | Standard (15-20 kg·cm) | -90° to +90° (180°) | 0.0° (Neutral pronation) |
| **CH 5** | `joint_6_gripper` | Micro/Standard (5-15 kg·cm) | 0% (Close) to 100% (Open) | 100% (Fully open) |
| **CH 6-10**| `finger_thumb` .. `pinky` | Micro Linear/Rotary (optional) | 0% to 100% per finger | 100% (Fully open) |

### C. Biosignal Acquisition Hardware Interface (Serial DAQ)
- **Port**: `/dev/ttyACM0` or `/dev/ttyUSB0` (Baud: `115200`)
- **EMG Channels** (4 Surface Differential Channels):
  - Ch 0: Forearm Flexor (*Flexor Digitorum Superficialis*) -> Grasp Close / Proportional Force
  - Ch 1: Forearm Extensor (*Extensor Digitorum*) -> Hand Open / Release
  - Ch 2: Biceps Brachii -> Arm Flexion / Reach trigger
  - Ch 3: Triceps Brachii -> Arm Extension
- **EEG Channels** (4 Channels, 10-20 International System):
  - Ch 0: **C3** (Left Sensorimotor Cortex - Right hand motor imagery)
  - Ch 1: **C4** (Right Sensorimotor Cortex - Left hand motor imagery)
  - Ch 2: **Cz** (Vertex / Midline - Reach planning & integration)
  - Ch 3: **Fz / Fp1** (Frontal - Cognitive attention & eye blink artifact trigger)
  - Reference: Earlobes (A1/A2) or Mastoid

---

## 3. Mathematical Formulations & Coordinate Transformations

### A. Coordinate Frames
1. **Camera Optical Frame** ($C$):
   - $+X_C$: Right
   - $+Y_C$: Down
   - $+Z_C$: Forward (Depth vector into scene)
2. **Robot Arm Base Frame** ($B$):
   - $+X_B$: Forward (Towards front workspace)
   - $+Y_B$: Left
   - $+Z_B$: Up (Vertical axis of base swivel)

### B. 3D Camera Deprojection & Base Transformation
Given pixel $(u, v)$ with depth $Z_C$ (in meters) and camera intrinsics $(f_x, f_y, c_x, c_y)$:
$$X_C = \frac{(u - c_x) \cdot Z_C}{f_x}, \quad Y_C = \frac{(v - c_y) \cdot Z_C}{f_y}, \quad Z_C = \text{depth}(u, v)$$

Homogeneous coordinate transformation to Robot Base frame:
$$\begin{bmatrix} X_B \\ Y_B \\ Z_B \\ 1 \end{bmatrix} = \mathbf{T}_B^C \begin{bmatrix} X_C \\ Y_C \\ Z_C \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{R}_B^C & \mathbf{t}_B^C \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} X_C \\ Y_C \\ Z_C \\ 1 \end{bmatrix}$$

### C. Inverse Kinematics (6-DOF Articulated Arm)
1. **Base Yaw ($\theta_1$)**:
   $$\theta_1 = \text{atan2}(Y_B, X_B)$$
2. **Wrist Center Coordinates ($r_w, z_w$)**:
   $$r_{\text{total}} = \sqrt{X_B^2 + Y_B^2}$$
   $$r_w = r_{\text{total}} - (L_4 + L_5) \cos(\theta_{\text{pitch}})$$
   $$z_w = Z_B - L_1 - (L_4 + L_5) \sin(\theta_{\text{pitch}})$$
3. **Elbow Pitch ($\theta_3$) via Law of Cosines**:
   $$D = \frac{r_w^2 + z_w^2 - L_2^2 - L_3^2}{2 L_2 L_3}, \quad \theta_3 = \text{atan2}(-\sqrt{1 - D^2}, D)$$
4. **Shoulder Pitch ($\theta_2$)**:
   $$\theta_2 = \text{atan2}(z_w, r_w) - \text{atan2}(L_3 \sin(\theta_3), L_2 + L_3 \cos(\theta_3))$$
5. **Wrist Pitch ($\theta_4$)**:
   $$\theta_4 = \theta_{\text{pitch}} - (\theta_2 + \theta_3)$$

### D. Trajectory Interpolation (Minimum-Jerk Quintic Polynomial)
For normalized time $\tau = \frac{t}{T} \in [0, 1]$:
$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$$
$$q(t) = q_{\text{start}} + (q_{\text{target}} - q_{\text{start}}) \cdot s(\tau)$$
Ensures zero velocity and zero acceleration at trajectory boundaries to eliminate mechanical jerk.

---

## 4. Biosignal DSP & Decoding Pipeline

```
Raw EMG/EEG Stream (1000Hz / 250Hz)
       │
       ├─► 50Hz/60Hz IIR Notch Filter (Powerline noise elimination)
       │
       ├─► 4th-Order Butterworth Bandpass:
       │     - EMG: 20 Hz - 450 Hz
       │     - EEG: 1 Hz - 45 Hz (Mu: 8-12Hz, Beta: 13-30Hz)
       │
       ├─► EMG: Full-Wave Rectification + Moving RMS Window (50ms)
       │     - Normalized Muscle Activation $\in [0, 1]$
       │     - Proportional Grip Force: $F_{\text{target}} = F_{\min} + (F_{\max} - F_{\min}) \cdot \text{Act}_{\text{flexor}}$
       │     - Co-Contraction Detection ($\text{Act}_{\text{flex}} > 0.85 \land \text{Act}_{\text{ext}} > 0.85 \implies \text{E-STOP}$)
       │
       └─► EEG: Fast Fourier Transform (FFT) Power Spectral Density
             - Event-Related Desynchronization (ERD) in Mu band: $\text{ERD} = \frac{P_{\text{rest}} - P_{\text{active}}}{P_{\text{rest}}}$
             - $\text{ERD} > 0.35 \implies \text{Motor Imagery Trigger (Reach)}$
             - Attention Index: $\frac{P_{\text{beta}}}{P_{\text{mu}} + \epsilon} > 0.60 \implies \text{Target Lock Confirmation}$
```

---

## 5. System State Machine Flow

| State | Entry Condition | Active Processes | Next State Transitions |
| :--- | :--- | :--- | :--- |
| **`IDLE`** | Startup or Return Home | Arm at home, camera active | $\to$ `SCANNING` |
| **`SCANNING`** | Visible objects found | RealSense 3D tracking & ranking | $\to$ `TARGET_SELECTED` (EEG Cycle / Gaze) |
| **`TARGET_SELECTED`** | Target locked | Calculates 3D approach & target force | $\to$ `PLANNING` (EEG Reach / Muscle flex) |
| **`PLANNING`** | Target confirmed | IK solver + Trajectory generation | $\to$ `REACHING` (if IK feasible) |
| **`REACHING`** | Trajectory ready | 50Hz waypoint streaming, pre-opens hand | $\to$ `AT_TARGET` (Reach complete) |
| **`AT_TARGET`** | Reached grasp point | Waits for grasp trigger | $\to$ `GRASPING` (EMG flexor burst) |
| **`GRASPING`** | Grasp triggered | Closed-loop force ramp with current feedback | $\to$ `HOLDING` (Target force reached) |
| **`HOLDING`** | Object secured | Proportional force modulation | $\to$ `RELEASING` (EMG extensor burst) |
| **`RELEASING`** | Release triggered | Opens gripper to 100%, retracts arm | $\to$ `IDLE` (Returns home) |
| **`EMERGENCY_STOP`** | Co-contraction / Limit violation | Immediate PWM kill / torque freeze | $\to$ `IDLE` (Spacebar / Manual reset) |

---

## 6. Directory Structure & Key Code References

- [`architecture.md`](file:///home/iedc_ai_dgx1/Documents/VAPA/architecture.md): Complete circuit diagrams, power architecture, and wiring specs.
- [`firmware/esp32_sensor_node/src/main.cpp`](file:///home/iedc_ai_dgx1/Documents/VAPA/firmware/esp32_sensor_node/src/main.cpp): ESP32 C++ firmware (Single ADS1115 + 5x FSRs + AS5600 Encoders + 100Hz JSON Serial2).
- [`drivers/pca9685_actuator.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/drivers/pca9685_actuator.py): VAPAActuatorController (5x MG996R Fingers CH 0-4, 3x DS3225 Arm CH 5-7, 1x DS3218 Wrist CH 8).
- [`drivers/pca9685_12ch_driver.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/drivers/pca9685_12ch_driver.py): 12-Channel Servo Driver with soft-start velocity curves.
- [`drivers/tca9548a_as5600.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/drivers/tca9548a_as5600.py): TCA9548A I2C Multiplexer & 4x AS5600 Magnetic Encoder driver.
- [`biosignals/esp32_serial_receiver.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/biosignals/esp32_serial_receiver.py): High-speed async UART telemetry receiver for ESP32 (5x FSRs, EMG, EEG, Encoders).
- [`config/system_config.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/config/system_config.py): Global dimensions, limits, filter thresholds.
- [`config/hardware_config.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/config/hardware_config.py): Jetson Orin pinout, PCA9685 servo channel map, I2C addresses, UART specs.
- [`vision/realsense_camera.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/vision/realsense_camera.py): RealSense RGB-D capture + Synthetic simulation fallback.
- [`vision/object_detector.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/vision/object_detector.py): Multi-backend object detector (MediaPipe, YOLO, 3D Depth Segmenter).
- [`vision/spatial_3d.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/vision/spatial_3d.py): 3D deprojection, base frame transform, grasp pose planning.
- [`vision/visualizer_3d.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/vision/visualizer_3d.py): OpenCV 3D bounding HUD and depth colormap renderer.
- [`biosignals/signal_filters.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/biosignals/signal_filters.py): Notch, Butterworth bandpass, RMS envelope, FFT bandpower.
- [`biosignals/emg_decoder.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/biosignals/emg_decoder.py): Muscle activation, gesture intent, proportional force.
- [`biosignals/eeg_decoder.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/biosignals/eeg_decoder.py): Motor imagery (Mu/Beta ERD), attention scoring, target cycling.
- [`biosignals/biosignal_streamer.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/biosignals/biosignal_streamer.py): Serial DAQ interface + Synthetic signal generator.
- [`biosignals/intent_fusion.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/biosignals/intent_fusion.py): Multimodal neural-muscular-vision arbitrator.
- [`kinematics/arm_model.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/kinematics/arm_model.py): Link dimensions, joint limits, workspace bounds.
- [`kinematics/forward_kinematics.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/kinematics/forward_kinematics.py): End-effector and 3D joint skeleton positions.
- [`kinematics/inverse_kinematics.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/kinematics/inverse_kinematics.py): Analytical geometric and numerical DLS IK solvers.
- [`kinematics/trajectory_planner.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/kinematics/trajectory_planner.py): Minimum-jerk smooth joint trajectory interpolation.
- [`actuation/pca9685_controller.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/actuation/pca9685_controller.py): Jetson Orin 16-channel 12-bit I2C PWM driver.
- [`actuation/serial_servo_controller.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/actuation/serial_servo_controller.py): Smart bus servo UART interface.
- [`actuation/mock_arm_controller.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/actuation/mock_arm_controller.py): Virtual hardware simulation with current/force feedback.
- [`actuation/arm_controller.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/actuation/arm_controller.py): Multi-joint arm & multi-finger hand manager.
- [`core/state_machine.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/core/state_machine.py): System state manager.
- [`core/vapa_engine.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/core/vapa_engine.py): Multi-threaded orchestrator.
- [`vapa_app.py`](file:///home/iedc_ai_dgx1/Documents/VAPA/vapa_app.py): Master application with interactive HUD dashboard.
