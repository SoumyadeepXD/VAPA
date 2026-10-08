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
 |         │         ├── CH 0 : Thumb Flex (MG996R)                                                      |
 |         │         ├── CH 1 : Index Flex (MG996R)                                                      |
 |         │         ├── CH 2 : Middle Flex (MG996R)                                                     |
 |         │         ├── CH 3 : Ring Flex (MG996R)                                                       |
 |         │         ├── CH 4 : Pinky Flex (MG996R)                                                      |
 |         │         ├── CH 5 : Arm Base Yaw (DS3225 25kg)                                               |
 |         │         ├── CH 6 : Arm Shoulder Pitch (DS3225 25kg)                                         |
 |         │         ├── CH 7 : Arm Elbow Pitch (DS3225 25kg)                                            |
 |         │         └── CH 8 : Wrist Pitch (DS3218 20kg)                                                |
 |         │                                                                                             |
 |         └─► [ TCA9548A 8-Channel I2C Multiplexer @ 0x70 ] (Optional Aux Sensor MUX)                   |
 |                                                                                                       |
 |  [ Hardware UART (/dev/ttyTHS1) ] ◄════════════════════════════════════════════════════════════════╗  |
 +----------------------------------------------------------------------------------------------------║--+
                                              INTER-NODE UART LINK                                    ║
                                      (115200 Baud, 8-N-1, 100 Hz JSON Stream)                        ║
 +----------------------------------------------------------------------------------------------------║--+
 |  [ Serial2 (GPIO 16 RX / GPIO 17 TX) ] ════════════════════════════════════════════════════════════╝  |
 |                                                                                                       |
 |  [ ESP32 Dedicated Analog Pins (ADC1) ]                                                               |
 |         ├── GPIO 32 ◄── FSR 1: Thumb Contact Force                                                    |
 |         ├── GPIO 33 ◄── FSR 2: Index Contact Force                                                    |
 |         ├── GPIO 34 ◄── FSR 3: Middle Contact Force                                                   |
 |         ├── GPIO 35 ◄── FSR 4: Ring Contact Force                                                     |
 |         └── GPIO 36 ◄── FSR 5: Pinky Contact Force                                                    |
 |                                                                                                       |
 |  [ ESP32 Hardware I2C (GPIO 21 SDA / GPIO 22 SCL @ 400 kHz) ]                                         |
 |         ├── ADS1115 ADC @ 0x48 (16-bit, 860 SPS)                                                      |
 |         │     ├── CH A0 : MyoWare EMG Muscle Signal                                                   |
 |         │     └── CH A1 : EEG Analog Brainwave Signal                                                 |
 |         └── AS5600 12-bit Magnetic Rotary Encoder @ 0x36 (Raw angle 0 - 360°)                         |
 |                                                                                                       |
 |                                  NODE 2: ESP32 MICROCONTROLLER                                        |
 |                         (Multi-Sensor Node: Single ADS1115 + 5x FSRs + AS5600)                        |
 +-------------------------------------------------------------------------------------------------------+
```

---

## 3. Circuit Schematics & Pinout Tables

### A. Jetson Orin 40-Pin Header Connections
| Jetson Pin | Signal Name | Target Device | Target Pin | Function / Description |
| :--- | :--- | :--- | :--- | :--- |
| **Pin 1** | 3.3V VDD | PCA9685 & I2C Modules | VCC / VDD | Logic Power (3.3V) |
| **Pin 3** | I2C1_SDA | PCA9685 & I2C Modules | SDA | Main I2C Data (`/dev/i2c-1`) |
| **Pin 5** | I2C1_SCL | PCA9685 & I2C Modules | SCL | Main I2C Clock (`/dev/i2c-1`) |
| **Pin 6** | GND | Central Star Ground | GND | System Common Ground |
| **Pin 8** | UART1_TXD | ESP32 | GPIO 16 (RX2) | Jetson Transmit -> ESP32 Receive |
| **Pin 10** | UART1_RXD | ESP32 | GPIO 17 (TX2) | ESP32 Transmit -> Jetson Receive |

### B. PCA9685 Actuator Channel Allocation (`drivers/pca9685_actuator.py`)
| Channel | Joint / Actuator | Servo Model | Voltage | Max Torque | Range | Function |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CH 0** | `finger_thumb` | **MG996R** | 6.0V | 11 kg·cm | 0° - 180° | Thumb Flexion / Close |
| **CH 1** | `finger_index` | **MG996R** | 6.0V | 11 kg·cm | 0° - 180° | Index Finger Flexion |
| **CH 2** | `finger_middle`| **MG996R** | 6.0V | 11 kg·cm | 0° - 180° | Middle Finger Flexion |
| **CH 3** | `finger_ring` | **MG996R** | 6.0V | 11 kg·cm | 0° - 180° | Ring Finger Flexion |
| **CH 4** | `finger_pinky` | **MG996R** | 6.0V | 11 kg·cm | 0° - 180° | Little Finger Flexion |
| **CH 5** | `arm_base_yaw` | **DS3225** | 6.0V | 25 kg·cm | 0° - 180° | Arm Base Waist Swivel |
| **CH 6** | `arm_shoulder_pitch`| **DS3225**| 6.0V | 25 kg·cm | 0° - 180° | Shoulder Elevation / Lift |
| **CH 7** | `arm_elbow_pitch` | **DS3225** | 6.0V | 25 kg·cm | 0° - 180° | Forearm Flexion / Extension |
| **CH 8** | `wrist_pitch` | **DS3218** | 6.0V | 20 kg·cm | 0° - 180° | Wrist Pitch (Flex / Extend) |

### C. ESP32 Node 2 Pinouts & Multi-Sensor Mapping (`src/main.cpp`)
```
 ESP32 DevKit V1 Pinouts:
 ├── VIN ────────◄ 5.0V from BEC Module
 ├── GND ────────◄ Central Star Ground
 ├── GPIO 16 (RX2) ◄── Jetson Header Pin 8 (UART1_TXD)
 ├── GPIO 17 (TX2) ──► Jetson Header Pin 10 (UART1_RXD)
 ├── GPIO 21 (SDA) ──► I2C Bus Data (ADS1115 @ 0x48 + AS5600 @ 0x36)
 ├── GPIO 22 (SCL) ──► I2C Bus Clock
 │
 ├── GPIO 32 ◄── FSR 1: Thumb Contact Force
 ├── GPIO 33 ◄── FSR 2: Index Contact Force
 ├── GPIO 34 ◄── FSR 3: Middle Contact Force
 ├── GPIO 35 ◄── FSR 4: Ring Contact Force
 └── GPIO 36 ◄── FSR 5: Pinky Contact Force

 ADS1115 ADC Module (Address 0x48):
 ├── A0 ◄── MyoWare EMG Sensor Output
 └── A1 ◄── EEG Brainwave Sensor Output

 AS5600 12-bit Magnetic Rotary Encoder (Address 0x36):
 └── Directly polled over Wire (Registers 0x0C/0x0D -> 0.0° - 360.0°)
```

---

## 4. Inter-Node Protocol & JSON Framing

The ESP32 continuously transmits JSON telemetry frames to the Jetson Orin over **Serial2 (`115200 Baud`) at 100 Hz (every 10ms)**:

```json
{"seq":1425,"fsr":[0.420,0.850,0.120,0.050,0.030],"emg":0.940,"eeg":0.315,"enc":[142.5],"ts":482910}
```

### JSON Schema Breakdown:
- **`seq`** (*uint32*): Monotonically increasing packet sequence counter for packet drop detection.
- **`fsr`** (*array of 5 floats*): Filtered voltages ($0.000\text{V} - 3.300\text{V}$) corresponding to Thumb, Index, Middle, Ring, Pinky fingertip forces.
- **`emg`** (*float*): Filtered MyoWare EMG muscle envelope voltage ($0.000\text{V} - 3.300\text{V}$).
- **`eeg`** (*float*): Filtered EEG brainwave analog input voltage.
- **`enc`** (*array of floats*): Live AS5600 magnetic rotary encoder angle in degrees ($0.0° - 360.0°$).
- **`ts`** (*uint32*): ESP32 internal millisecond timestamp (`millis()`).

---

## 5. Electrical Engineering & Signal Conditioning Mathematics

This section provides the rigorous circuit equations, component ratings, and digital signal transformations governing VAPA's hardware nodes.

```
 [ 3S LiPo 11.1V-12.6V ] ──► [ Buck 6V @ 10-15A ] ──► [ 4700µF Low-ESR Decoupling ] ──► PCA9685 Servos
                                                                                               │
 [ FSR 402 + 10kΩ ] ────► [ RC LPF (fc = 159Hz) ] ──► [ ADS1115 16-bit ADC ] ──► EMA Filter ──► ESP32 UART
                                                                                               │
 [ Jetson Orin /dev/i2c-1 ] ────► [ Prescale = 121 (50Hz) ] ──► [ 12-bit Tick Mapping ] ───────┘
```

### A. Power Budget, Battery Sizing & Decoupling Capacitor Derivation

#### 1. 3S LiPo Battery Discharge Model
* **Operating Range**:
  $$V_{\text{nominal}} = 3 \times 3.7\text{ V} = 11.1\text{ V}, \quad V_{\max} = 3 \times 4.2\text{ V} = 12.6\text{ V}, \quad V_{\text{cutoff}} = 3 \times 3.2\text{ V} = 9.6\text{ V}$$
* **Discharge Capability**:
  For a $2200\text{ mAh}$, $30\text{C}$ pack:
  $$I_{\text{continuous\_max}} = C_{\text{rate}} \times \text{Capacity} = 30\text{ h}^{-1} \times 2.2\text{ Ah} = 66.0\text{ A}$$
  Protected downstream by a $30\text{ A}$ automotive blade fuse ($I_{\text{fuse}} < \frac{1}{2} I_{\text{continuous\_max}}$).

#### 2. System Power Dissipation
* **Jetson Orin (15W power profile)**: $P_{\text{Jetson}} \approx 15.0\text{ W} \implies I_{12\text{V}} \approx 1.25\text{ A}$
* **Servos Peak Draw (worst-case multi-joint lift)**:
  $$P_{\text{servos}} = V_{\text{rail}} \cdot I_{\text{peak}} = 6.0\text{ V} \times 12.0\text{ A} = 72.0\text{ W}$$
* **Logic BEC & Microcontrollers**: $P_{\text{logic}} \approx 5.0\text{ V} \times 0.6\text{ A} = 3.0\text{ W}$
* **Peak Power Sum**:
  $$P_{\text{peak}} \approx 15.0\text{ W} + 72.0\text{ W} + 3.0\text{ W} = 90.0\text{ W} \quad (\approx 8.1\text{ A @ } 11.1\text{ V})$$
* **Buck Converter Thermal Loss** ($\eta \approx 92\%$ efficiency):
  $$P_{\text{loss}} = P_{\text{out}} \left( \frac{1 - \eta}{\eta} \right) = 72.0\text{ W} \times \left(\frac{0.08}{0.92}\right) = 6.26\text{ W}$$

#### 3. Servo Inrush Decoupling Capacitor Sizing ($4700\mu\text{F}$)
Simultaneous acceleration of heavy DS3225 joints pulls step current transients $\Delta I = 10.0\text{ A}$ across buck switching loop response time $\Delta t \approx 2.0\text{ ms}$. To guarantee supply droop $\Delta V_{\text{ripple}} \le 0.5\text{ V}$ (preventing servo controller resets and logic brownouts):

$$C \ge \frac{\Delta I \cdot \Delta t}{\Delta V_{\text{ripple}}} = \frac{10.0\text{ A} \times 2.0 \times 10^{-3}\text{ s}}{0.5\text{ V}} = 4.0 \times 10^{-3}\text{ F} = 4000\,\mu\text{F}$$

*Component Selection*: **$4700\,\mu\text{F} / 16\text{V}$ Low-ESR ($< 25\text{ m}\Omega$) Electrolytic Capacitor** wired directly across the PCA9685 $V_+$ and GND screw terminals.

---

### B. Sensor Signal Conditioning & ADC Transfer Functions

#### 1. ADS1115 16-Bit Sigma-Delta ADC Resolution
Configured with internal Programmable Gain Amplifier (PGA) at `GAIN_ONE` ($\pm 4.096\text{ V}$ Full-Scale Range):

$$\text{LSB Resolution} = \frac{V_{\text{FSR}}}{2^{15} - 1} = \frac{4.096\text{ V}}{32767} = 1.25 \times 10^{-4}\text{ V} = 125.0\,\mu\text{V/bit}$$

Voltage conversion formula implemented in firmware:

$$V_{\text{measured}} = \max\left(0, N_{\text{raw}}\right) \times 0.000125\text{ V}$$

#### 2. FSR 402 Fingertip Force Sensor Circuit
Each FSR forms a voltage divider with reference $V_{\text{ref}} = 3.3\text{ V}$ and pull-down resistor $R_{\text{fixed}} = 10\text{ k}\Omega$:

$$V_{\text{out}} = V_{\text{ref}} \cdot \frac{R_{\text{fixed}}}{R_{\text{FSR}}(F) + R_{\text{fixed}}}$$

Inverting for the instantaneous piezoresistive sensor resistance $R_{\text{FSR}}$:

$$R_{\text{FSR}} = R_{\text{fixed}} \left( \frac{V_{\text{ref}}}{V_{\text{out}}} - 1 \right) = 10000 \cdot \left( \frac{3.3}{V_{\text{out}}} - 1 \right)\,\Omega$$

Applied normal force scales as $F \propto R_{\text{FSR}}^{-\gamma} \approx \left( \frac{V_{\text{out}}}{V_{\text{ref}} - V_{\text{out}}} \right)^{1/\gamma}$ (with empirical exponent $\gamma \approx 0.85$).

#### 3. Analog RC Anti-Aliasing Filter
Each analog input includes a parallel $C = 100\text{ nF}$ capacitor across $R_{\text{fixed}} = 10\text{ k}\Omega$. Under high sensor resistance (low force):

$$f_{c, \text{RC}} = \frac{1}{2\pi R_{\text{fixed}} C} = \frac{1}{2\pi (10 \times 10^3\,\Omega)(100 \times 10^{-9}\text{ F})} = \frac{1}{2\pi \times 10^{-3}} \approx 159.15\text{ Hz}$$

Because the ADS1115 operates at $860\text{ SPS}$ ($f_{\text{Nyquist}} = 430\text{ Hz}$), high-frequency motor noise above $159\text{ Hz}$ is effectively suppressed before digital sampling.

#### 4. Digital Exponential Moving Average (EMA) Low-Pass Filter
The ESP32 firmware executes real-time discrete low-pass smoothing at loop rate $f_s = 100\text{ Hz}$ ($\Delta t = 10\text{ ms}$):

$$y[k] = \alpha \cdot x[k] + (1 - \alpha) \cdot y[k - 1]$$

Equivalent continuous -3dB cutoff frequency $f_{c, \text{digital}}$:

$$f_{c, \text{digital}} = \frac{\alpha}{2\pi \Delta t (1 - \alpha)}$$

| Channel | Alpha ($\alpha$) | Sampling Rate ($f_s$) | Effective Cutoff ($f_c$) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **FSR Tactile** | $0.35$ | $100\text{ Hz}$ | $\approx 8.57\text{ Hz}$ | Rapid contact onset detection without mechanical bounce |
| **EMG Envelope** | $0.25$ | $100\text{ Hz}$ | $\approx 5.31\text{ Hz}$ | Reconstructs smooth muscular activation envelope |
| **EEG Rhythms** | $0.20$ | $100\text{ Hz}$ | $\approx 3.98\text{ Hz}$ | Attenuates high-frequency baseline drift and RF pickup |

---

### C. PCA9685 PWM Timing & Microsecond Mapping

#### 1. Prescaler Register Derivation
The PCA9685 operates with a $25\text{ MHz}$ internal clock. The 8-bit prescaler determines output frequency $f_{\text{pwm}}$:

$$\text{PRESCALE} = \text{round}\left( \frac{f_{\text{osc}}}{4096 \times f_{\text{pwm}}} \right) - 1$$

For $f_{\text{pwm}} = 50\text{ Hz}$ ($T_{\text{period}} = 20\text{ ms} = 20,000\,\mu\text{s}$):

$$\text{PRESCALE} = \text{round}\left( \frac{25 \times 10^6}{4096 \times 50} \right) - 1 = \text{round}(122.07) - 1 = 121 = \text{0x79}$$

#### 2. PWM Resolution & Microsecond-to-Tick Mapping
Each 12-bit tick period is:

$$\Delta t_{\text{tick}} = \frac{T_{\text{period}}}{4096} = \frac{20000\,\mu\text{s}}{4096} \approx 4.8828\,\mu\text{s/tick}$$

To command target joint angle $\theta \in [\theta_{\min}, \theta_{\max}]$:

$$t_{\mu\text{s}}(\theta) = t_{\min} + \left( \frac{(\theta + \theta_{\text{trim}}) \cdot \text{dir} - \theta_{\min}}{\theta_{\max} - \theta_{\min}} \right) \cdot (t_{\max} - t_{\min})$$

$$\text{Tick}_{\text{off}} = \text{round}\left( \frac{t_{\mu\text{s}}(\theta)}{20000\,\mu\text{s}} \times 4096 \right) = \text{round}\left( t_{\mu\text{s}}(\theta) \times 0.2048 \right)$$

*Example*: Neutral servo position ($1500\,\mu\text{s}$) produces $\text{Tick}_{\text{off}} = \text{round}(1500 \times 0.2048) = 307$.

---

### D. AS5600 12-Bit Magnetic Rotary Encoder Angular Equations

The AS5600 measures diametric magnet orientation via integrated Hall sensors:
* **Resolution**: 12-bit ($N_{\text{total}} = 2^{12} = 4096$ counts per $360^\circ$).
* **Angular LSB**:
  $$\Delta \theta = \frac{360.0^\circ}{4096} \approx 0.08789^\circ/\text{LSB} = 1.534 \times 10^{-3}\text{ rad}$$
* **Calibrated Angle Transformation**:
  $$\theta_{\text{raw}} = \left( \frac{N_{\text{raw}}}{4096.0} \right) \times 360.0^\circ$$
  $$\theta_{\text{calibrated}} = \left[ (\theta_{\text{raw}} \cdot \text{dir}) - \theta_{\text{zero\_offset}} \right] \pmod{360.0^\circ}$$

---

### E. Quintic S-Curve Soft-Start Velocity Profile

To eliminate current spikes from high-torque servos (DS3225), transitions follow a jerk-free quintic profile evaluated at $50\text{ Hz}$ ($\Delta t = 20\text{ ms}$):

$$\tau = \frac{t}{T} \in [0, 1]$$

$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$$

Velocity $\dot{s}(\tau)$ and acceleration $\ddot{s}(\tau)$ derivatives:

$$\dot{s}(\tau) = \frac{1}{T} \left( 30\tau^2 - 60\tau^3 + 30\tau^4 \right)$$

$$\ddot{s}(\tau) = \frac{1}{T^2} \left( 60\tau - 180\tau^2 + 120\tau^3 \right)$$

* **Peak Velocity**: Occurs at midpoint $\tau = 0.5$:
  $$\dot{s}(0.5) = \frac{1}{T} (30 \cdot 0.25 - 60 \cdot 0.125 + 30 \cdot 0.0625) = \frac{1.875}{T}$$
  $$v_{\max} = \frac{1.875}{T} \cdot |\theta_{\text{target}} - \theta_{\text{start}}| \le v_{\text{rated}}$$
* **Zero Boundary Conditions**:
  $$\dot{s}(0) = \dot{s}(1) = 0, \quad \ddot{s}(0) = \ddot{s}(1) = 0$$

guaranteeing zero torque discontinuity and protecting servo gear trains from shock wear.

---

## 6. Complete Repository Folder Structure

```
VAPA/
├── config/
│   ├── __init__.py                # Package initialization for config
│   ├── system_config.py           # Global kinematic dimensions, limits, filter thresholds
│   └── hardware_config.py         # Jetson Orin pinouts, 12-servo map, I2C addresses, UART specs
├── firmware/
│   └── esp32_sensor_node/
│       ├── src/
│       │   └── main.cpp           # ESP32 C++ PlatformIO firmware (Single ADS1115 + 5x FSR + AS5600)
│       ├── esp32_sensor_node.ino  # ESP32 Arduino sketch (Single ADS1115 + 5x FSR + AS5600)
│       └── platformio.ini         # PlatformIO build configuration & library dependencies
├── drivers/
│   ├── __init__.py                # Package initialization for hardware drivers
│   ├── pca9685_actuator.py        # VAPAActuatorController: 5x MG996R Fingers + 3x DS3225 Arm + 1x DS3218 Wrist
│   ├── pca9685_12ch_driver.py     # 12-Channel Servo Driver with soft-start S-curve interpolation
│   └── tca9548a_as5600.py         # TCA9548A I2C Multiplexer & 4x AS5600 Magnetic Encoder driver
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

## 7. Software Dependencies & Installation

### A. NVIDIA Jetson Orin (Python 3)
Install Python dependencies into virtual environment:
```bash
pip install smbus2 adafruit-circuitpython-pca9685 pyserial pyrealsense2 opencv-python numpy scipy scikit-learn
```

### B. ESP32 Arduino / PlatformIO Libraries
- **`Adafruit ADS1X15`** (v2.4.2+)
- **`Adafruit BusIO`** (v1.16.1+)
- **`Wire`** (Built-in ESP32 I2C)
