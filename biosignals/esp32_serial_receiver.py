"""
VAPA Asynchronous ESP32 Serial Telemetry Receiver
High-speed non-blocking UART receiver on NVIDIA Jetson Orin (auto-probes /dev/ttyTHS1, /dev/ttyTHS0, /dev/ttyUSB0, /dev/ttyACM0).
Ingests, deserializes, and validates real-time 100 Hz JSON frames from ESP32 Node 2:
- 5x FSR 402 Tactile Sensors (Thumb, Index, Middle, Ring, Little)
- MyoWare 2.0 EMG Muscle Signal (ADS1115 A0)
- EEG Brainwave Analog Signal (ADS1115 A1)
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
    FSR_CHANNELS,
)

logger = logging.getLogger("VAPA.Biosignals.ESP32Receiver")

# FSR 402 Calibration Constant: Converts 0-3.3V divider voltage to Newtons
# With 10k divider to GND: V_out = 3.3 * (10k / (R_fsr + 10k))
# Higher pressure -> lower R_fsr -> higher V_out
FSR_VOLTAGE_TO_FORCE_FACTOR = 3.5  # Newtons per Volt


class ESP32TelemetryFrame:
    """Represents a validated timestamped sensor packet from ESP32."""
    __slots__ = (
        "seq",
        "fsr_volts",
        "fsr_forces_n",
        "emg_volts",
        "emg_activation",
        "eeg_volts",
        "enc_deg",
        "esp_timestamp_ms",
        "arrival_time",
        "is_valid",
    )

    def __init__(
        self,
        seq: int = 0,
        fsr_volts: list[float] = None,
        emg_volts: float = 0.0,
        eeg_volts: float = 0.0,
        enc_deg: list[float] = None,
        esp_timestamp_ms: int = 0,
        is_valid: bool = True,
    ):
        self.seq = int(seq)
        # 5 FSRs: [Thumb, Index, Middle, Ring, Pinky]
        self.fsr_volts = fsr_volts if (fsr_volts and len(fsr_volts) == 5) else [0.0, 0.0, 0.0, 0.0, 0.0]
        # Calculate contact force in Newtons
        self.fsr_forces_n = [float(max(0.0, v * FSR_VOLTAGE_TO_FORCE_FACTOR)) for v in self.fsr_volts]
        self.emg_volts = float(emg_volts)
        # Normalize EMG voltage (0.15V baseline to 2.5V full flex -> 0.0 to 1.0 activation)
        self.emg_activation = float(np.clip((self.emg_volts - 0.15) / 2.2, 0.0, 1.0))
        self.eeg_volts = float(eeg_volts)
        self.enc_deg = enc_deg or [0.0]
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
            f"EMG_Act={self.emg_activation*100:.0f}%, EEG_V={self.eeg_volts:.2f}V, Enc={enc_val:.1f}°)"
        )


class AsyncESP32Receiver:
    """
    Non-blocking background thread that consumes serial telemetry from ESP32.
    Auto-probes /dev/ttyTHS1, /dev/ttyTHS0, /dev/ttyUSB0, /dev/ttyACM0.
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

        if not self.force_mock and SERIAL_AVAILABLE:
            self._connect()

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

        for port_candidate in candidate_ports:
            if not os.path.exists(port_candidate):
                continue

            try:
                conn = serial.Serial(port_candidate, self.baud_rate, timeout=0.15)
                conn.reset_input_buffer()
                self.serial_conn = conn
                self.port = port_candidate
                self.is_connected = True
                logger.info(f"[SUCCESS] Connected to ESP32 UART on {port_candidate} @ {self.baud_rate} Baud.")
                return True
            except Exception as e:
                logger.debug(f"Port {port_candidate} could not be opened: {e}")
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

    def _worker_loop(self):
        """Continuous background loop deserializing incoming lines at 100 Hz."""
        while self.running:
            if not self.is_connected or self.serial_conn is None or not self.serial_conn.is_open:
                if not self.force_mock and SERIAL_AVAILABLE:
                    time.sleep(2.0)
                    self._connect()
                    continue
                else:
                    self._generate_mock_frame()
                    time.sleep(0.01)  # 100 Hz mock
                    continue

            try:
                line_bytes = self.serial_conn.readline()
                if not line_bytes:
                    continue

                line_str = line_bytes.decode("utf-8", errors="ignore").strip()
                if not (line_str.startswith("{") and line_str.endswith("}")):
                    continue

                # Parse JSON packet
                data = json.loads(line_str)

                seq = data.get("seq", 0)
                fsr = data.get("fsr", [0.0, 0.0, 0.0, 0.0, 0.0])
                emg = data.get("emg", 0.0)
                eeg = data.get("eeg", 0.0)
                enc = data.get("enc", [0.0])
                ts  = data.get("ts", 0)

                # Ensure fsr has 5 values
                if len(fsr) < 5:
                    fsr = fsr + [0.0] * (5 - len(fsr))
                elif len(fsr) > 5:
                    fsr = fsr[:5]

                if self.last_seq != -1 and seq > (self.last_seq + 1):
                    dropped = seq - (self.last_seq + 1)
                    self.dropped_count += dropped

                self.last_seq = seq
                self.packet_count += 1

                frame = ESP32TelemetryFrame(
                    seq=seq,
                    fsr_volts=fsr,
                    emg_volts=emg,
                    eeg_volts=eeg,
                    enc_deg=enc,
                    esp_timestamp_ms=ts,
                    is_valid=True,
                )

                with self.lock:
                    self.latest_frame = frame

            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading ESP32 UART packet: {e}")
                time.sleep(0.05)

    def _generate_mock_frame(self):
        """Generates realistic synthetic telemetry when physical UART is disconnected."""
        t = time.time()
        self.packet_count += 1
        seq = self.packet_count

        # Simulated baseline for 5x FSRs (Thumb, Index, Middle, Ring, Pinky)
        sim_fsr = [
            max(0.0, float(0.05 + 0.03 * np.sin(t * 2.0))),
            max(0.0, float(0.04 + 0.03 * np.sin(t * 2.0 + 0.5))),
            max(0.0, float(0.03 + 0.02 * np.sin(t * 2.0 + 1.0))),
            max(0.0, float(0.02 + 0.02 * np.sin(t * 2.0 + 1.5))),
            max(0.0, float(0.01 + 0.01 * np.sin(t * 2.0 + 2.0))),
        ]
        sim_emg = float(0.20 + 0.05 * np.sin(t * 0.8))
        sim_eeg = float(0.35 + 0.10 * np.sin(t * 10.0 * 2 * np.pi))
        sim_enc = [float((t * 20.0) % 360.0)]

        frame = ESP32TelemetryFrame(
            seq=seq,
            fsr_volts=sim_fsr,
            emg_volts=sim_emg,
            eeg_volts=sim_eeg,
            enc_deg=sim_enc,
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
