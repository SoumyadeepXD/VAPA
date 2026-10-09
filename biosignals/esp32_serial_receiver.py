"""
VAPA Asynchronous ESP32 Serial Telemetry Receiver
High-speed non-blocking UART receiver on NVIDIA Jetson Orin (auto-probes /dev/ttyTHS1, /dev/ttyTHS0, /dev/ttyUSB0, /dev/ttyACM0).
Ingests, deserializes, and validates real-time 100 Hz binary / JSON frames from ESP32 Node 2:
- 5x FSR 402 Tactile Sensors (Thumb, Index, Middle, Ring, Little)
- Two-Site EMG: Flexor (ADS1115 #1 A0) & Extensor (ADS1115 #2 A2)
- EEG Brainwave Analog Signal (ADS1115 #1 A1)
- 12-bit AS5600 Magnetic Rotary Encoder Angle
- Hardware Emergency Stop Button Sense & Jetson Heartbeat Guard
"""

import os
import time
import json
import logging
import threading
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from config.hardware_config import (
    BIOSIGNAL_SERIAL_PORT,
    BIOSIGNAL_BAUD_RATE,
    JETSON_UART_CANDIDATES,
    JETSON_UART_BAUD_CANDIDATES,
    FSR_CHANNELS,
)

logger = logging.getLogger("VAPA.Biosignals.ESP32Receiver")

# FSR 402 Calibration Constant: Converts 0-3.3V divider voltage to Newtons
# With 10k divider to GND: V_out = 3.3 * (10k / (R_fsr + 10k))
# Higher pressure -> lower R_fsr -> higher V_out
FSR_VOLTAGE_TO_FORCE_FACTOR = 3.5  # Newtons per Volt


def compute_crc16(data: bytes | bytearray) -> int:
    """Computes CRC-16-CCITT (False: poly 0x1021, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class ESP32TelemetryFrame:
    """Represents a validated timestamped sensor packet from ESP32."""
    __slots__ = (
        "seq",
        "fsr_volts",
        "fsr_forces_n",
        "emg_flex_volts",
        "emg_ext_volts",
        "emg_flex_activation",
        "emg_ext_activation",
        "emg_volts",
        "emg_activation",
        "eeg_volts",
        "enc_deg",
        "estop_button_pressed",
        "esp_timestamp_ms",
        "arrival_time",
        "is_valid",
    )

    def __init__(
        self,
        seq: int = 0,
        fsr_volts: list[float] = None,
        emg_volts: float = None,
        emg_flex_volts: float = 0.0,
        emg_ext_volts: float = 0.0,
        eeg_volts: float = 0.0,
        enc_deg: list[float] = None,
        estop_button_pressed: bool = False,
        esp_timestamp_ms: int = 0,
        is_valid: bool = True,
    ):
        self.seq = int(seq)
        # 5 FSRs: [Thumb, Index, Middle, Ring, Pinky]
        self.fsr_volts = fsr_volts if (fsr_volts and len(fsr_volts) == 5) else [0.0, 0.0, 0.0, 0.0, 0.0]
        # Calculate contact force in Newtons
        self.fsr_forces_n = [float(max(0.0, v * FSR_VOLTAGE_TO_FORCE_FACTOR)) for v in self.fsr_volts]
        
        # Two-channel EMG handling with backward compatibility
        if emg_volts is not None and emg_flex_volts == 0.0:
            self.emg_flex_volts = float(emg_volts)
        else:
            self.emg_flex_volts = float(emg_flex_volts)
        self.emg_ext_volts = float(emg_ext_volts)
        self.emg_volts = self.emg_flex_volts

        # Proportional activation normalization (0.15V baseline to 2.35V peak -> 0.0 to 1.0 activation)
        self.emg_flex_activation = float(np.clip((self.emg_flex_volts - 0.15) / 2.2, 0.0, 1.0))
        self.emg_ext_activation = float(np.clip((self.emg_ext_volts - 0.15) / 2.2, 0.0, 1.0))
        self.emg_activation = self.emg_flex_activation

        self.eeg_volts = float(eeg_volts)
        self.enc_deg = enc_deg or [0.0]
        self.estop_button_pressed = bool(estop_button_pressed)
        self.esp_timestamp_ms = int(esp_timestamp_ms)
        self.arrival_time = time.time()
        self.is_valid = is_valid

    @property
    def total_grip_force_n(self) -> float:
        return sum(self.fsr_forces_n)

    def get_finger_force_map(self) -> dict[str, float]:
        names = ["thumb", "index", "middle", "ring", "pinky"]
        return {names[i]: self.fsr_forces_n[i] for i in range(min(5, len(self.fsr_forces_n)))}

    @property
    def encoder_angle_deg(self) -> float:
        """Returns primary AS5600 magnetic encoder angle in degrees."""
        return self.enc_deg[0] if self.enc_deg else 0.0

    def __repr__(self):
        f = self.fsr_forces_n
        enc_val = self.enc_deg[0] if self.enc_deg else 0.0
        fsr_str = f"Th:{f[0]:.1f}, In:{f[1]:.1f}, Mi:{f[2]:.1f}, Ri:{f[3]:.1f}"
        if len(f) > 4:
            fsr_str += f", Pi:{f[4]:.1f}"
        return (
            f"ESP32Frame(seq={self.seq}, FSR_N=[{fsr_str}], "
            f"Flex={self.emg_flex_activation*100:.0f}%, Ext={self.emg_ext_activation*100:.0f}%, "
            f"EEG_V={self.eeg_volts:.2f}V, Enc={enc_val:.1f}°, EStop={self.estop_button_pressed})"
        )


class AsyncESP32Receiver:
    """
    Non-blocking background thread that consumes serial telemetry from ESP32.
    Supports both 33-byte CRC16 binary framing and 100 Hz JSON streaming.
    Transmits 20 Hz Jetson heartbeat to keep ESP32 PCA9685 OE active.
    """
    def __init__(self, port: str = BIOSIGNAL_SERIAL_PORT, baud_rate: int = BIOSIGNAL_BAUD_RATE, force_mock: bool = False):
        self.port = port
        self.baud_rate = baud_rate
        self.force_mock = force_mock
        self.serial_conn = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Telemetry State
        self.latest_frame = ESP32TelemetryFrame(is_valid=False)
        self.packet_count = 0
        self.dropped_count = 0
        self.last_seq = -1
        self.is_connected = False

        # Heartbeat & Framing
        self._last_heartbeat_time = 0.0

        # Mock Scenario Generator State
        self._mock_scenario = "auto"
        self._mock_override_scenario = None
        self._mock_override_until = 0.0

        if not self.force_mock and SERIAL_AVAILABLE:
            self._connect()

    def set_mock_scenario(self, scenario: str):
        """Sets persistent mock scenario: 'rest', 'flexor', 'extensor', 'co_contraction', 'auto'."""
        with self.lock:
            self._mock_scenario = scenario

    def trigger_synthetic_gesture(self, gesture_name: str, duration_s: float = 1.0):
        """Injects a temporary mock scenario for duration_s seconds."""
        with self.lock:
            self._mock_override_scenario = gesture_name
            self._mock_override_until = time.time() + duration_s
        logger.info(f"Triggered synthetic gesture on ESP32 receiver: {gesture_name} for {duration_s:.1f}s")

    def _connect(self) -> bool:
        """Attempts to discover and open UART serial connection to ESP32."""
        if not SERIAL_AVAILABLE:
            logger.warning("pyserial is not installed. Running in mock biosignal mode.")
            self.is_connected = False
            return False

        candidate_ports = [self.port]
        for p in JETSON_UART_CANDIDATES:
            if p not in candidate_ports:
                candidate_ports.append(p)

        # Also probe any /dev/ttyUSB* or /dev/ttyACM*
        try:
            for fname in os.listdir("/dev"):
                full_path = f"/dev/{fname}"
                if (fname.startswith("ttyTHS") or fname.startswith("ttyUSB") or fname.startswith("ttyACM")) and full_path not in candidate_ports:
                    candidate_ports.append(full_path)
        except Exception:
            pass

        baud_candidates = [self.baud_rate]
        for b in JETSON_UART_BAUD_CANDIDATES:
            if b not in baud_candidates:
                baud_candidates.append(b)

        for port_candidate in candidate_ports:
            if not os.path.exists(port_candidate):
                continue

            for baud in baud_candidates:
                try:
                    conn = serial.Serial(port_candidate, baud, timeout=0.10)
                    conn.reset_input_buffer()
                    self.serial_conn = conn
                    self.port = port_candidate
                    self.baud_rate = baud
                    self.is_connected = True
                    logger.info(f"[SUCCESS] Connected to ESP32 UART on {port_candidate} @ {self.baud_rate} Baud.")
                    return True
                except Exception as e:
                    logger.debug(f"Port {port_candidate} @ {baud} could not be opened: {e}")
                    continue

        logger.warning(
            f"Could not open ESP32 on ports {candidate_ports}. "
            f"Check wiring: Jetson Pin 8 (TXD) -> ESP32 GPIO16, Pin 10 (RXD) <- ESP32 GPIO17, Pin 9 -> GND. "
            f"Also verify user in dialout group ('sudo usermod -a -G dialout $USER'). Falling back to simulation."
        )
        self.is_connected = False
        return False

    def start(self):
        """Starts the background receiver thread."""
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, name="ESP32_Receiver_Thread", daemon=True)
        self.thread.start()
        logger.info("AsyncESP32Receiver background thread started.")

    def stop(self):
        """Stops the receiver thread and closes serial port."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        logger.info("AsyncESP32Receiver stopped.")

    def _record_frame(self, frame: ESP32TelemetryFrame):
        if self.last_seq != -1 and frame.seq > (self.last_seq + 1):
            dropped = frame.seq - (self.last_seq + 1)
            self.dropped_count += dropped

        self.last_seq = frame.seq
        self.packet_count += 1
        with self.lock:
            self.latest_frame = frame

    def _worker_loop(self):
        """Continuous background loop deserializing incoming frames at 100 Hz."""
        rx_buf = bytearray()

        while self.running:
            now = time.time()

            # 1. Jetson-to-ESP32 20 Hz Heartbeat Transmission
            if self.is_connected and self.serial_conn and self.serial_conn.is_open:
                if (now - self._last_heartbeat_time) >= 0.05:  # 20 Hz
                    try:
                        self.serial_conn.write(b'H')
                        self._last_heartbeat_time = now
                    except Exception:
                        pass

            # 2. Check Connection
            if not self.is_connected or self.serial_conn is None or not self.serial_conn.is_open:
                if not self.force_mock and SERIAL_AVAILABLE:
                    time.sleep(2.0)
                    self._connect()
                    continue
                else:
                    self._generate_mock_frame()
                    time.sleep(0.01)  # 100 Hz mock
                    continue

            # 3. Read Incoming Bytes
            try:
                in_waiting = self.serial_conn.in_waiting
                if in_waiting > 0:
                    chunk = self.serial_conn.read(in_waiting)
                    if chunk:
                        rx_buf.extend(chunk)
                else:
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"Error reading ESP32 UART: {e}")
                time.sleep(0.05)
                continue

            # 4. Parse Frames from Buffer
            while len(rx_buf) > 0:
                # Mode A: Compact Binary Frame (Header: 0xAA 0x55, Length: 33)
                if rx_buf[0] == 0xAA:
                    if len(rx_buf) < 2:
                        break
                    if rx_buf[1] == 0x55:
                        if len(rx_buf) < 33:
                            break  # Wait for full packet
                        pkt = rx_buf[:33]
                        expected_crc = (pkt[30] << 8) | pkt[31]
                        actual_crc = compute_crc16(pkt[2:30])
                        if actual_crc == expected_crc:
                            seq = (pkt[3] << 24) | (pkt[4] << 16) | (pkt[5] << 8) | pkt[6]
                            fsrs = [((pkt[7 + i*2] << 8) | pkt[8 + i*2]) / 1000.0 for i in range(5)]
                            flex = ((pkt[17] << 8) | pkt[18]) / 1000.0
                            ext = ((pkt[19] << 8) | pkt[20]) / 1000.0
                            eeg = ((pkt[21] << 8) | pkt[22]) / 1000.0
                            enc = [((pkt[23] << 8) | pkt[24]) / 100.0]
                            estop = bool(pkt[25] != 0)
                            ts = (pkt[26] << 24) | (pkt[27] << 16) | (pkt[28] << 8) | pkt[29]

                            frame = ESP32TelemetryFrame(
                                seq=seq,
                                fsr_volts=fsrs,
                                emg_flex_volts=flex,
                                emg_ext_volts=ext,
                                eeg_volts=eeg,
                                enc_deg=enc,
                                estop_button_pressed=estop,
                                esp_timestamp_ms=ts,
                                is_valid=True,
                            )
                            self._record_frame(frame)
                            del rx_buf[:33]
                            continue
                        else:
                            # CRC Mismatch -> advance 1 byte to resync
                            self.dropped_count += 1
                            del rx_buf[:1]
                            continue
                    else:
                        del rx_buf[:1]
                        continue

                # Mode B: High-Speed JSON Frame (Starts with '{', Ends with '\n')
                elif rx_buf[0] == ord('{'):
                    nl_pos = rx_buf.find(b'\n')
                    if nl_pos == -1:
                        if len(rx_buf) > 512:
                            del rx_buf[:1]
                        break
                    line_bytes = rx_buf[:nl_pos+1]
                    del rx_buf[:nl_pos+1]
                    try:
                        line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                        data = json.loads(line_str)
                        seq = data.get("seq", 0)
                        fsr = data.get("fsr", [0.0, 0.0, 0.0, 0.0, 0.0])
                        if len(fsr) < 5:
                            fsr = fsr + [0.0] * (5 - len(fsr))
                        elif len(fsr) > 5:
                            fsr = fsr[:5]
                        emg_flex = data.get("emg_flex", data.get("emg", 0.0))
                        emg_ext = data.get("emg_ext", 0.0)
                        eeg = data.get("eeg", 0.0)
                        enc = data.get("enc", [0.0])
                        estop = bool(data.get("estop", 0))
                        ts = data.get("ts", 0)

                        frame = ESP32TelemetryFrame(
                            seq=seq,
                            fsr_volts=fsr,
                            emg_flex_volts=emg_flex,
                            emg_ext_volts=emg_ext,
                            eeg_volts=eeg,
                            enc_deg=enc,
                            estop_button_pressed=estop,
                            esp_timestamp_ms=ts,
                            is_valid=True,
                        )
                        self._record_frame(frame)
                    except Exception:
                        self.dropped_count += 1
                    continue

                else:
                    # Seek next candidate header
                    p_aa = rx_buf.find(b'\xaa')
                    p_br = rx_buf.find(b'{')
                    if p_aa != -1 and p_br != -1:
                        del rx_buf[:min(p_aa, p_br)]
                    elif p_aa != -1:
                        del rx_buf[:p_aa]
                    elif p_br != -1:
                        del rx_buf[:p_br]
                    else:
                        rx_buf.clear()
                    break

    def _generate_mock_frame(self):
        """Generates realistic synthetic telemetry when physical UART is disconnected."""
        t = time.time()
        self.packet_count += 1
        seq = self.packet_count

        with self.lock:
            if t < self._mock_override_until and self._mock_override_scenario:
                scenario = self._mock_override_scenario
            else:
                scenario = self._mock_scenario

        # Default baselines
        sim_fsr = [0.02, 0.02, 0.02, 0.02, 0.02]
        sim_flex = 0.08 + 0.02 * np.sin(t * 1.5)
        sim_ext  = 0.07 + 0.02 * np.cos(t * 1.5)
        sim_eeg  = 0.35 + 0.10 * np.sin(t * 10.0 * 2 * np.pi)
        sim_enc  = [float((t * 20.0) % 360.0)]
        sim_estop = False

        if scenario == "rest":
            sim_flex = 0.08 + 0.01 * np.sin(t * 1.5)
            sim_ext  = 0.07 + 0.01 * np.cos(t * 1.5)
            sim_fsr  = [0.01, 0.01, 0.01, 0.01, 0.01]

        elif scenario in ("flexor", "flex", "grasp", "fist"):
            # High flexor activation (> 0.85), low extensor
            sim_flex = float(2.15 + 0.05 * np.sin(t * 3.0))
            sim_ext  = float(0.08 + 0.02 * np.cos(t * 1.5))
            sim_fsr  = [0.65, 0.85, 0.70, 0.50, 0.30]

        elif scenario in ("extensor", "ext", "open"):
            # High extensor activation (> 0.85), low flexor
            sim_flex = float(0.08 + 0.02 * np.cos(t * 1.5))
            sim_ext  = float(2.15 + 0.05 * np.sin(t * 3.0))
            sim_fsr  = [0.02, 0.02, 0.02, 0.02, 0.02]

        elif scenario in ("co_contraction", "co_contract", "estop"):
            # Simultaneous flexor > 0.85 and extensor > 0.85 -> triggers E-Stop!
            sim_flex = float(2.20 + 0.05 * np.sin(t * 4.0))
            sim_ext  = float(2.20 + 0.05 * np.cos(t * 4.0))

        elif scenario == "estop_button":
            sim_estop = True

        elif scenario == "auto":
            # Dynamic gentle sine wave
            sim_fsr = [
                max(0.0, float(0.05 + 0.03 * np.sin(t * 2.0))),
                max(0.0, float(0.04 + 0.03 * np.sin(t * 2.0 + 0.5))),
                max(0.0, float(0.03 + 0.02 * np.sin(t * 2.0 + 1.0))),
                max(0.0, float(0.02 + 0.02 * np.sin(t * 2.0 + 1.5))),
                max(0.0, float(0.01 + 0.01 * np.sin(t * 2.0 + 2.0))),
            ]
            sim_flex = float(0.20 + 0.05 * np.sin(t * 0.8))
            sim_ext  = float(0.18 + 0.04 * np.cos(t * 0.8))

        frame = ESP32TelemetryFrame(
            seq=seq,
            fsr_volts=sim_fsr,
            emg_flex_volts=sim_flex,
            emg_ext_volts=sim_ext,
            eeg_volts=sim_eeg,
            enc_deg=sim_enc,
            estop_button_pressed=sim_estop,
            esp_timestamp_ms=int(t * 1000) % 1000000,
            is_valid=True,
        )

        with self.lock:
            self.latest_frame = frame

    def get_latest_frame(self) -> ESP32TelemetryFrame:
        with self.lock:
            return self.latest_frame

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "packets_received": self.packet_count,
                "packets_dropped": self.dropped_count,
                "is_connected": self.is_connected,
                "port": self.port,
            }


# Canonical alias for telemetry receiver
ESP32TelemetryReceiver = AsyncESP32Receiver
