

/*
VAPA - Multi-Sensor Node (Single ADS1115 + 5x FSRs + AS5600 Encoders)
- 1x ADS1115 (0x48): A0 = MyoWare EMG, A1 = EEG Output
- 5x FSR Sensors on ESP32 Analog Pins (GPIO 32, 33, 34, 35, 36)
- AS5600 Magnetic Encoders (I2C Address 0x36)
- UART2 Output to Jetson Orin @ 115200 Baud (100 Hz JSON)
*/

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ==============================================================================
// PIN DEFINITIONS & CONSTANTS
// ==============================================================================
#define I2C_SDA_PIN         21
#define I2C_SCL_PIN         22
#define I2C_CLOCK_FREQ_HZ   400000  // Fast I2C 400 kHz

#define UART2_RX_PIN        16
#define UART2_TX_PIN        17
#define UART2_BAUD_RATE     115200

#define STATUS_LED_PIN      2

// 5x FSR Analog Pins
#define FSR1_PIN            32  // Thumb
#define FSR2_PIN            33  // Index
#define FSR3_PIN            34  // Middle
#define FSR4_PIN            35  // Ring
#define FSR5_PIN            36  // Pinky

#define TARGET_SAMPLE_PERIOD_US 10000 // 100 Hz loop timing

// I2C Addresses
#define ADS1115_BIO_ADDR    0x48
#define AS5600_I2C_ADDR     0x36
#define AS5600_RAW_ANGLE_REG 0x0C

#define ADS_VOLTS_PER_BIT   0.000125f

// Exponential Moving Average (EMA) Alpha values
#define EMA_ALPHA_FSR       0.35f
#define EMA_ALPHA_EMG       0.25f
#define EMA_ALPHA_EEG       0.20f
#define EMA_ALPHA_ENC       0.30f

// ==============================================================================
// GLOBAL OBJECTS & STATE
// ==============================================================================
Adafruit_ADS1115 ads_bio;
bool ads_bio_connected = false;
bool as5600_connected  = false;

float filtered_fsr[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
float filtered_emg    = 0.0f;
float filtered_eeg    = 0.0f;
float filtered_enc1   = 0.0f; // Single or main encoder angle in degrees

uint32_t packet_seq = 0;
uint32_t last_sample_time_us = 0;
uint32_t last_heartbeat_ms = 0;
bool led_state = false;

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

// Direct 12-bit raw angle read from AS5600 register over Wire
uint16_t read_as5600_raw_angle() {
  Wire.beginTransmission(AS5600_I2C_ADDR);
  Wire.write(AS5600_RAW_ANGLE_REG);
  if (Wire.endTransmission() != 0) {
    return 0; // Read error or not connected
  }
  
  Wire.requestFrom(AS5600_I2C_ADDR, 2);
  if (Wire.available() >= 2) {
    uint16_t high_byte = Wire.read();
    uint16_t low_byte  = Wire.read();
    return ((high_byte << 8) | low_byte) & 0x0FFF; // 12-bit mask (0-4095)
  }
  return 0;
}

// ==============================================================================
// SETUP
// ==============================================================================
void setup() {
  Serial.begin(115200);
  Serial2.begin(UART2_BAUD_RATE, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH);

  analogSetAttenuation(ADC_11db);
  pinMode(FSR1_PIN, INPUT);
  pinMode(FSR2_PIN, INPUT);
  pinMode(FSR3_PIN, INPUT);
  pinMode(FSR4_PIN, INPUT);
  pinMode(FSR5_PIN, INPUT);

  delay(100);
  Serial.println("\n[INFO] Starting VAPA Sensor Node (ADS1115 + 5xFSR + AS5600)...");

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_CLOCK_FREQ_HZ);

  // Initialize ADS1115
  if (ads_bio.begin(ADS1115_BIO_ADDR)) {
      ads_bio.setGain(GAIN_ONE);
      ads_bio.setDataRate(RATE_ADS1115_860SPS);
      ads_bio_connected = true;
      Serial.println("[OK] ADS1115 (0x48) Initialized.");
  } else {
      Serial.println("[WARN] ADS1115 at 0x48 not found!");
  }

  // Check AS5600 presence on I2C bus
  Wire.beginTransmission(AS5600_I2C_ADDR);
  if (Wire.endTransmission() == 0) {
      as5600_connected = true;
      Serial.println("[OK] AS5600 Encoder (0x36) Initialized.");
  } else {
      Serial.println("[WARN] AS5600 Encoder at 0x36 not detected.");
  }

  digitalWrite(STATUS_LED_PIN, LOW);
  last_sample_time_us = micros();
}

// ==============================================================================
// MAIN LOOP (100 Hz)
// ==============================================================================
void loop() {
  uint32_t current_time_us = micros();

  if ((current_time_us - last_sample_time_us) >= TARGET_SAMPLE_PERIOD_US) {
      last_sample_time_us += TARGET_SAMPLE_PERIOD_US;

      // 1. Read 5x FSRs from ESP32 Analog Pins
      int raw_fsr[5];
      raw_fsr[0] = analogRead(FSR1_PIN);
      raw_fsr[1] = analogRead(FSR2_PIN);
      raw_fsr[2] = analogRead(FSR3_PIN);
      raw_fsr[3] = analogRead(FSR4_PIN);
      raw_fsr[4] = analogRead(FSR5_PIN);

      for (int i = 0; i < 5; i++) {
          float v_fsr = ((float)raw_fsr[i] / 4095.0f) * 3.3f;
          filtered_fsr[i] = apply_ema(v_fsr, filtered_fsr[i], EMA_ALPHA_FSR);
      }

      // 2. Read Biosignals from ADS1115
      int16_t raw_emg = 0;
      int16_t raw_eeg = 0;
      if (ads_bio_connected) {
          raw_emg = ads_bio.readADC_SingleEnded(0); // A0: MyoWare EMG
          raw_eeg = ads_bio.readADC_SingleEnded(1); // A1: EEG Output
      }

      float v_emg = adc_to_voltage(raw_emg);
      float v_eeg = adc_to_voltage(raw_eeg);
      filtered_emg = apply_ema(v_emg, filtered_emg, EMA_ALPHA_EMG);
      filtered_eeg = apply_ema(v_eeg, filtered_eeg, EMA_ALPHA_EEG);

      // 3. Read AS5600 Encoder Angle
      float raw_deg = 0.0f;
      if (as5600_connected) {
          uint16_t raw_angle = read_as5600_raw_angle();
          raw_deg = ((float)raw_angle / 4095.0f) * 360.0f;
      }
      filtered_enc1 = apply_ema(raw_deg, filtered_enc1, EMA_ALPHA_ENC);

      // 4. Send JSON Payload over Serial2
      // Format: {"seq":100,"fsr":[0.1,0.2,0.0,0.0,0.0],"emg":0.52,"eeg":0.12,"enc":[142.5],"ts":10020}
      char json_buffer[200];
      snprintf(json_buffer, sizeof(json_buffer),
               "{\"seq\":%lu,\"fsr\":[%.3f,%.3f,%.3f,%.3f,%.3f],\"emg\":%.3f,\"eeg\":%.3f,\"enc\":[%.1f],\"ts\":%lu}\n",
               packet_seq++,
               filtered_fsr[0], filtered_fsr[1], filtered_fsr[2], filtered_fsr[3], filtered_fsr[4],
               filtered_emg,
               filtered_eeg,
               filtered_enc1,
               millis());

      Serial2.print(json_buffer);

      if (packet_seq % 50 == 0) {
          Serial.print("[TX] ");
          Serial.print(json_buffer);
      }
  }

  // Heartbeat LED (1 Hz)
  if (millis() - last_heartbeat_ms >= 500) {
      last_heartbeat_ms = millis();
      led_state = !led_state;
      digitalWrite(STATUS_LED_PIN, led_state ? HIGH : LOW);
  }
}
