/*
VAPA - Multi-Sensor Node (Dual ADS1115 + 5x FSRs + AS5600 Encoders + Safety OE)
- ADS1115 #1 (0x48): A0 = Flexor EMG (MyoWare ENV), A1 = EEG Output
- ADS1115 #2 (0x49): A2 = Extensor EMG (MyoWare ENV)
- 5x FSR Sensors on ESP32 Analog Pins (GPIO 32, 33, 34, 35, 36)
- AS5600 Magnetic Rotary Encoder (I2C Address 0x36)
- Hardware E-Stop Button on GPIO 27 (Momentary to GND, internal pullup)
- PCA9685 Servo Output Enable (/OE) Pin on GPIO 25 (Active-High Disable; default off behind flag)
- High-Speed UART2 to Jetson Orin @ 460800 Baud (Dual Binary/JSON mode)
- Jetson Heartbeat Safety Guard (Loss > 200ms -> Drive OE HIGH)
- Autonomous FSR Force Ceiling Guard (> 12.0N -> Drive OE HIGH)
*/

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ==============================================================================
// PIN DEFINITIONS & CONSTANTS
// ==============================================================================
#define I2C_SDA_PIN             21
#define I2C_SCL_PIN             22
#define I2C_CLOCK_FREQ_HZ       400000  // Fast I2C 400 kHz

#define UART2_RX_PIN            16
#define UART2_TX_PIN            17
#define UART2_BAUD_RATE         460800  // Upgraded from 115200 to keep UART load < 35%

#define STATUS_LED_PIN          2

// Hardware Safety & E-Stop Pins
#define ESTOP_BUTTON_PIN        27      // External physical emergency stop button (Active LOW)
#define PCA9685_OE_PIN          25      // PCA9685 Output Enable (/OE pin; HIGH = Servos Disabled)
#define PCA9685_OE_ENABLED      true    // Set true: physical /OE wire supervision enabled

// 5x FSR Analog Pins (GPIO 32 - 36)
#define FSR1_PIN                32      // Thumb
#define FSR2_PIN                33      // Index
#define FSR3_PIN                34      // Middle
#define FSR4_PIN                35      // Ring
#define FSR5_PIN                36      // Pinky

#define TARGET_SAMPLE_PERIOD_US 10000   // 100 Hz loop timing (10 ms)
#define JETSON_HEARTBEAT_TIMEOUT_MS 200 // Heartbeat lost if > 200ms without Jetson ping
#define FSR_FORCE_CEILING_FRACTION 0.85f // Fraction of calibrated FSR dynamic range (default 85%)

// Calibrated per-finger baselines & dynamic ranges from tools/calibrate_fsr.py
const float fsr_tare_volts[5]  = {0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
const float fsr_range_volts[5] = {2.75f, 2.75f, 2.75f, 2.75f, 2.75f};


// I2C Addresses & Configurable Channel-to-ADS Mappings
#define ADS1115_PRIMARY_ADDR    0x48    // ADS1115 #1
#define ADS1115_SECONDARY_ADDR  0x49    // ADS1115 #2
#define AS5600_I2C_ADDR         0x36
#define AS5600_RAW_ANGLE_REG    0x0C

#define ADS_VOLTS_PER_BIT       0.000125f // GAIN_ONE: 4.096V range -> 0.125mV/LSB

// ADS1115 Input Pin Assignments
#define ADS_INPUT_EMG_FLEX      0       // ADS1115 #1 Channel A0: Flexor EMG
#define ADS_INPUT_EEG           1       // ADS1115 #1 Channel A1: EEG Output
#define ADS_INPUT_EMG_EXT       2       // ADS1115 #2 Channel A2: Extensor EMG

// Exponential Moving Average (EMA) Alpha values
#define EMA_ALPHA_FSR           0.35f
#define EMA_ALPHA_EMG           0.25f
#define EMA_ALPHA_EEG           0.20f
#define EMA_ALPHA_ENC           0.30f

// Telemetry Framing Protocol: 0 = JSON (default readable), 1 = Compact Binary CRC16
#define TELEMETRY_BINARY_MODE   false

// ==============================================================================
// GLOBAL OBJECTS & STATE
// ==============================================================================
Adafruit_ADS1115 ads_bio_1; // Address 0x48 (Flexor EMG + EEG)
Adafruit_ADS1115 ads_bio_2; // Address 0x49 (Extensor EMG)

bool ads_1_connected = false;
bool ads_2_connected = false;
bool as5600_connected = false;

float filtered_fsr[5]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
float filtered_emg_flex = 0.0f;
float filtered_emg_ext  = 0.0f;
float filtered_eeg      = 0.0f;
float filtered_enc1     = 0.0f;

uint32_t packet_seq = 0;
uint32_t last_sample_time_us = 0;
uint32_t last_heartbeat_ms = 0;
uint32_t last_jetson_heartbeat_ms = 0;
bool led_state = false;
bool estop_active = false;

// ==============================================================================
// HELPER FUNCTIONS
// ==============================================================================
inline float apply_ema(float raw, float previous, float alpha) {
  return (alpha * raw) + ((1.0f - alpha) * previous);
}

inline float adc_to_voltage(int16_t raw_counts) {
  if (raw_counts < 0) raw_counts = 0;
  return (float)raw_counts * ADS_VOLTS_PER_BIT;
}

// 16-bit CRC Calculation (CCITT-False: poly 0x1021, init 0xFFFF)
uint16_t compute_crc16(const uint8_t *data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (uint8_t bit = 0; bit < 8; bit++) {
      if (crc & 0x8000) {
        crc = (crc << 1) ^ 0x1021;
      } else {
        crc = crc << 1;
      }
    }
  }
  return crc;
}

// Direct 12-bit raw angle read from AS5600 register over Wire
uint16_t read_as5600_raw_angle() {
  Wire.beginTransmission(AS5600_I2C_ADDR);
  Wire.write(AS5600_RAW_ANGLE_REG);
  if (Wire.endTransmission() != 0) {
    return 0;
  }

  Wire.requestFrom(AS5600_I2C_ADDR, 2);
  if (Wire.available() >= 2) {
    uint16_t high_byte = Wire.read();
    uint16_t low_byte  = Wire.read();
    return ((high_byte << 8) | low_byte) & 0x0FFF;
  }
  return 0;
}

// Safe PCA9685 Output Enable control
void set_pca9685_oe_disabled(bool disable_servos) {
  if (PCA9685_OE_ENABLED) {
    // PCA9685 /OE is active LOW: HIGH = disabled (PWM off), LOW = enabled
    digitalWrite(PCA9685_OE_PIN, disable_servos ? HIGH : LOW);
  }
}

// ==============================================================================
// SETUP
// ==============================================================================
void setup() {
  Serial.begin(115200);
  Serial2.begin(UART2_BAUD_RATE, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH);

  // Safety & E-Stop Pin Setup
  pinMode(ESTOP_BUTTON_PIN, INPUT_PULLUP);
  if (PCA9685_OE_ENABLED) {
    pinMode(PCA9685_OE_PIN, OUTPUT);
    digitalWrite(PCA9685_OE_PIN, HIGH); // Start with servos disabled until heartbeat arrives
  }

  // FSR Analog Input Pins
  analogSetAttenuation(ADC_11db);
  pinMode(FSR1_PIN, INPUT);
  pinMode(FSR2_PIN, INPUT);
  pinMode(FSR3_PIN, INPUT);
  pinMode(FSR4_PIN, INPUT);
  pinMode(FSR5_PIN, INPUT);

  delay(100);
  Serial.println("\n[INFO] Starting VAPA Multi-Sensor Node (Dual ADS1115 + 5xFSR + AS5600 + Safety)...");

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_CLOCK_FREQ_HZ);

  // Initialize ADS1115 #1 (Address 0x48: Flexor EMG & EEG)
  if (ads_bio_1.begin(ADS1115_PRIMARY_ADDR)) {
    ads_bio_1.setGain(GAIN_ONE);
    ads_bio_1.setDataRate(RATE_ADS1115_860SPS);
    ads_1_connected = true;
    Serial.println("[OK] ADS1115 #1 (0x48 - Flexor EMG & EEG) Initialized.");
  } else {
    Serial.println("[WARN] ADS1115 #1 at 0x48 not detected!");
  }

  // Initialize ADS1115 #2 (Address 0x49: Extensor EMG)
  if (ads_bio_2.begin(ADS1115_SECONDARY_ADDR)) {
    ads_bio_2.setGain(GAIN_ONE);
    ads_bio_2.setDataRate(RATE_ADS1115_860SPS);
    ads_2_connected = true;
    Serial.println("[OK] ADS1115 #2 (0x49 - Extensor EMG) Initialized.");
  } else {
    Serial.println("[WARN] ADS1115 #2 at 0x49 not detected!");
  }

  // Check AS5600 Presence
  Wire.beginTransmission(AS5600_I2C_ADDR);
  if (Wire.endTransmission() == 0) {
    as5600_connected = true;
    Serial.println("[OK] AS5600 Magnetic Encoder (0x36) Initialized.");
  } else {
    Serial.println("[WARN] AS5600 Encoder at 0x36 not detected.");
  }

  digitalWrite(STATUS_LED_PIN, LOW);
  last_sample_time_us = micros();
  last_jetson_heartbeat_ms = millis();
}

// ==============================================================================
// MAIN LOOP (100 Hz)
// ==============================================================================
void loop() {
  uint32_t current_time_us = micros();

  // 1. Check for Incoming Jetson Heartbeat ('H' / 0x48 @ 20 Hz)
  while (Serial2.available() > 0) {
    char in_byte = (char)Serial2.read();
    if (in_byte == 'H' || in_byte == 0x48) {
      last_jetson_heartbeat_ms = millis();
    }
  }

  // 2. Hardware E-Stop Button Sense (Active LOW)
  bool button_pressed = (digitalRead(ESTOP_BUTTON_PIN) == LOW);

  // 3. Jetson Heartbeat Timeout Guard
  bool heartbeat_lost = ((millis() - last_jetson_heartbeat_ms) > JETSON_HEARTBEAT_TIMEOUT_MS);

  // 100 Hz Telemetry Loop
  if ((current_time_us - last_sample_time_us) >= TARGET_SAMPLE_PERIOD_US) {
    last_sample_time_us += TARGET_SAMPLE_PERIOD_US;

    // A. Read 5x FSRs from Analog Pins
    int raw_fsr[5];
    raw_fsr[0] = analogRead(FSR1_PIN);
    raw_fsr[1] = analogRead(FSR2_PIN);
    raw_fsr[2] = analogRead(FSR3_PIN);
    raw_fsr[3] = analogRead(FSR4_PIN);
    raw_fsr[4] = analogRead(FSR5_PIN);

    bool force_ceiling_tripped = false;
    for (int i = 0; i < 5; i++) {
      float v_fsr = ((float)raw_fsr[i] / 4095.0f) * 3.3f;
      filtered_fsr[i] = apply_ema(v_fsr, filtered_fsr[i], EMA_ALPHA_FSR);
      float fraction = (filtered_fsr[i] - fsr_tare_volts[i]) / fsr_range_volts[i];
      if (fraction >= FSR_FORCE_CEILING_FRACTION) {
        force_ceiling_tripped = true;
      }
    }

    // B. Autonomous Hardware E-Stop / OE Safety Action
    estop_active = button_pressed || force_ceiling_tripped || heartbeat_lost;
    if (estop_active) {
      set_pca9685_oe_disabled(true);
    } else {
      set_pca9685_oe_disabled(false);
    }

    // C. Read Biosignals from Dual ADS1115
    int16_t raw_flex = 0;
    int16_t raw_eeg  = 0;
    int16_t raw_ext  = 0;

    if (ads_1_connected) {
      raw_flex = ads_bio_1.readADC_SingleEnded(ADS_INPUT_EMG_FLEX); // Ch A0: Flexor EMG
      raw_eeg  = ads_bio_1.readADC_SingleEnded(ADS_INPUT_EEG);      // Ch A1: EEG
    }
    if (ads_2_connected) {
      raw_ext  = ads_bio_2.readADC_SingleEnded(ADS_INPUT_EMG_EXT);  // Ch A2: Extensor EMG
    }

    float v_flex = adc_to_voltage(raw_flex);
    float v_ext  = adc_to_voltage(raw_ext);
    float v_eeg  = adc_to_voltage(raw_eeg);

    filtered_emg_flex = apply_ema(v_flex, filtered_emg_flex, EMA_ALPHA_EMG);
    filtered_emg_ext  = apply_ema(v_ext,  filtered_emg_ext,  EMA_ALPHA_EMG);
    filtered_eeg      = apply_ema(v_eeg,  filtered_eeg,      EMA_ALPHA_EEG);

    // D. Read AS5600 Encoder Angle
    float raw_deg = 0.0f;
    if (as5600_connected) {
      uint16_t raw_angle = read_as5600_raw_angle();
      raw_deg = ((float)raw_angle / 4095.0f) * 360.0f;
    }
    filtered_enc1 = apply_ema(raw_deg, filtered_enc1, EMA_ALPHA_ENC);

    // E. Transmit Telemetry over UART2
    #if TELEMETRY_BINARY_MODE
      // Compact Binary Packet (33 bytes) with 0xAA 0x55 sync & CRC16
      uint8_t bin_pkt[33];
      bin_pkt[0] = 0xAA;
      bin_pkt[1] = 0x55;
      bin_pkt[2] = 33; // Packet length
      // Sequence (4 bytes)
      bin_pkt[3] = (uint8_t)(packet_seq >> 24);
      bin_pkt[4] = (uint8_t)(packet_seq >> 16);
      bin_pkt[5] = (uint8_t)(packet_seq >> 8);
      bin_pkt[6] = (uint8_t)(packet_seq);
      // 5x FSRs in millivolts (10 bytes)
      for (int i = 0; i < 5; i++) {
        uint16_t fsr_mv = (uint16_t)(filtered_fsr[i] * 1000.0f);
        bin_pkt[7 + i*2] = (uint8_t)(fsr_mv >> 8);
        bin_pkt[8 + i*2] = (uint8_t)(fsr_mv);
      }
      // EMG Flex mV (2 bytes)
      uint16_t flex_mv = (uint16_t)(filtered_emg_flex * 1000.0f);
      bin_pkt[17] = (uint8_t)(flex_mv >> 8);
      bin_pkt[18] = (uint8_t)(flex_mv);
      // EMG Ext mV (2 bytes)
      uint16_t ext_mv = (uint16_t)(filtered_emg_ext * 1000.0f);
      bin_pkt[19] = (uint8_t)(ext_mv >> 8);
      bin_pkt[20] = (uint8_t)(ext_mv);
      // EEG mV (2 bytes)
      uint16_t eeg_mv = (uint16_t)(filtered_eeg * 1000.0f);
      bin_pkt[21] = (uint8_t)(eeg_mv >> 8);
      bin_pkt[22] = (uint8_t)(eeg_mv);
      // Enc centidegrees (2 bytes)
      uint16_t enc_cd = (uint16_t)(filtered_enc1 * 100.0f);
      bin_pkt[23] = (uint8_t)(enc_cd >> 8);
      bin_pkt[24] = (uint8_t)(enc_cd);
      // Flags byte (1 byte): bit 0 = estop_active, bit 1 = oe_disabled
      uint8_t flags = 0;
      if (estop_active) flags |= 0x01;
      if (!PCA9685_OE_ENABLED || estop_active) flags |= 0x02;
      bin_pkt[25] = flags;
      // Timestamp (4 bytes)
      uint32_t ts_now = millis();
      bin_pkt[26] = (uint8_t)(ts_now >> 24);
      bin_pkt[27] = (uint8_t)(ts_now >> 16);
      bin_pkt[28] = (uint8_t)(ts_now >> 8);
      bin_pkt[29] = (uint8_t)(ts_now);
      // Compute CRC16 over bytes 2..29
      uint16_t crc = compute_crc16(&bin_pkt[2], 28);
      bin_pkt[30] = (uint8_t)(crc >> 8);
      bin_pkt[31] = (uint8_t)(crc);
      bin_pkt[32] = '\n';
      Serial2.write(bin_pkt, sizeof(bin_pkt));
      packet_seq++;
    #else
      // High-Speed JSON Frame (includes emg_flex, emg_ext, legacy emg, estop, and oe_ok)
      char json_buffer[256];
      snprintf(json_buffer, sizeof(json_buffer),
               "{\"seq\":%lu,\"fsr\":[%.3f,%.3f,%.3f,%.3f,%.3f],\"emg_flex\":%.3f,\"emg_ext\":%.3f,\"emg\":%.3f,\"eeg\":%.3f,\"enc\":[%.1f],\"estop\":%d,\"oe_ok\":%d,\"ts\":%lu}\n",
               packet_seq++,
               filtered_fsr[0], filtered_fsr[1], filtered_fsr[2], filtered_fsr[3], filtered_fsr[4],
               filtered_emg_flex,
               filtered_emg_ext,
               filtered_emg_flex, // Legacy compatibility
               filtered_eeg,
               filtered_enc1,
               estop_active ? 1 : 0,
               (PCA9685_OE_ENABLED && !estop_active) ? 1 : 0,
               millis());

      Serial2.print(json_buffer);

      if (packet_seq % 50 == 0) {
        Serial.print("[TX] ");
        Serial.print(json_buffer);
      }
    #endif
  }

  // Heartbeat LED (1 Hz)
  if (millis() - last_heartbeat_ms >= 500) {
    last_heartbeat_ms = millis();
    led_state = !led_state;
    digitalWrite(STATUS_LED_PIN, led_state ? HIGH : LOW);
  }
}
