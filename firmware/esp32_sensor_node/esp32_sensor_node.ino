/*
================================================================================
 VAPA - Visually Assisted Prosthetic Arm
 NODE 2: ESP32 Bio-Signal & Tactile Sensor Acquisition Firmware
 Built according to VAPA Complete Hardware Connection Guide (Sections 3, 4, 5)
================================================================================
 Hardware Setup:
 - MCU: ESP32-WROOM-32 / NodeMCU / DevKit V1
 - Power: 5V Logic Rail to VIN, Star GND to GND, 3V3 powers sensors
 - I2C Bus: GPIO 21 (SDA), GPIO 22 (SCL) with 4.7k ohm pull-ups to 3.3V
 - ADC Module: 1x ADS1115 (Address 0x48, ADDR -> GND)
     * A0: MyoWare 2.0 EMG Sensor SIG pin (0 to 3.3V analog)
     * A1: EEG Sensor Module Analog OUT (0 to 3.3V analog)
     * A2: Spare (tied to GND)
     * A3: Spare (tied to GND)
 - FSR402 Fingertip Tactile Force Sensors (5 fingers, internal ESP32 12-bit ADC):
     * GPIO 32: FSR #1 (Thumb)   - 10k divider + 100nF cap to GND
     * GPIO 33: FSR #2 (Index)   - 10k divider + 100nF cap to GND
     * GPIO 34: FSR #3 (Middle)  - 10k divider + 100nF cap to GND
     * GPIO 35: FSR #4 (Ring)    - 10k divider + 100nF cap to GND
     * GPIO 36: FSR #5 (Little)  - 10k divider + 100nF cap to GND
 - Inter-Node Communication:
     * Serial2: GPIO 17 (TX2) -> Jetson Pin 10 (UART1_RXD) @ 115200 Baud
     * Serial2: GPIO 16 (RX2) <- Jetson Pin 8 (UART1_TXD) @ 115200 Baud
     * Star GND <-> Jetson Pin 9 (GND)
     * Serial (USB): Diagnostic monitoring & debug logs @ 115200 Baud
 - Sampling Rate: 100 Hz (10ms loop cycle)
================================================================================
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

#define UART2_RX_PIN        16      // Connected to Jetson TX (Pin 8 / UART1_TXD)
#define UART2_TX_PIN        17      // Connected to Jetson RX (Pin 10 / UART1_RXD)
#define UART2_BAUD_RATE     115200

#define STATUS_LED_PIN      2       // Built-in LED for heartbeat / diagnostics

// 5x FSR402 Analog Pins (ESP32 ADC1 channels)
#define FSR_PIN_THUMB       32
#define FSR_PIN_INDEX       33
#define FSR_PIN_MIDDLE      34
#define FSR_PIN_RING        35
#define FSR_PIN_LITTLE      36

// Sampling Period (100 Hz -> 10,000 microseconds)
#define TARGET_SAMPLE_PERIOD_US 10000

// Single ADS1115 I2C Address (Section 5A: ADDR -> GND)
#define ADS1115_I2C_ADDR    0x48

// ADC Voltage Scale: GAIN_ONE gives +/- 4.096V range (1 bit = 0.125mV)
#define ADS_VOLTS_PER_BIT   0.000125f

// Digital Low-Pass Exponential Moving Average (EMA) Alpha factors
#define EMA_ALPHA_FSR       0.35f   // Fast tactile response
#define EMA_ALPHA_EMG       0.25f   // Smooth muscle envelope
#define EMA_ALPHA_EEG       0.20f   // Filter high-frequency noise

// ==============================================================================
// GLOBAL OBJECTS & STATE
// ==============================================================================
Adafruit_ADS1115 ads;  // Single ADS1115 at 0x48
bool ads_connected = false;

// Filtered Signal Buffers
// 5 FSRs: [Thumb, Index, Middle, Ring, Little]
float filtered_fsr[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
float filtered_emg    = 0.0f;
float filtered_eeg    = 0.0f;

// Frame Counter & Timing
uint32_t packet_seq = 0;
uint32_t last_sample_time_us = 0;
uint32_t last_heartbeat_ms = 0;
bool led_state = false;

const int FSR_PINS[5] = {
    FSR_PIN_THUMB,
    FSR_PIN_INDEX,
    FSR_PIN_MIDDLE,
    FSR_PIN_RING,
    FSR_PIN_LITTLE
};

// ==============================================================================
// FILTER & VOLTAGE HELPERS
// ==============================================================================
inline float apply_ema(float raw, float previous, float alpha) {
    return (alpha * raw) + ((1.0f - alpha) * previous);
}

inline float esp32_adc_to_volts(int raw_adc) {
    // ESP32 12-bit ADC: 0..4095 corresponds to ~0.0V to 3.3V
    return ((float)raw_adc / 4095.0f) * 3.3f;
}

inline float ads_to_volts(int16_t raw_counts) {
    if (raw_counts < 0) raw_counts = 0;
    return (float)raw_counts * ADS_VOLTS_PER_BIT;
}

// ==============================================================================
// INITIALIZATION
// ==============================================================================
void setup() {
    // 1. Initialize Diagnostic USB Serial & Inter-Node Serial2 to Jetson
    Serial.begin(115200);
    Serial2.begin(UART2_BAUD_RATE, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, HIGH);

    // Configure ESP32 ADC pins for 5x FSRs (12-bit resolution, 0-3.3V attenuation)
    analogReadResolution(12);
    for (int i = 0; i < 5; i++) {
        pinMode(FSR_PINS[i], INPUT);
    }

    delay(100);
    Serial.println("\n========================================================");
    Serial.println(" VAPA Node 2: ESP32 Sensor Acquisition Node Online");
    Serial.println(" Hardware Guide: 5x FSR (GPIO32-36) + ADS1115 (0x48)");
    Serial.println(" Streaming JSON @ 100Hz -> Jetson Orin via Serial2");
    Serial.println("========================================================");

    // 2. Initialize Hardware I2C Master Bus
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_CLOCK_FREQ_HZ);

    // 3. Initialize ADS1115 (Section 5A: Address 0x48 for EMG + EEG)
    if (ads.begin(ADS1115_I2C_ADDR)) {
        ads.setGain(GAIN_ONE);                     // +/- 4.096V range
        ads.setDataRate(RATE_ADS1115_860SPS);       // Maximum speed (860 SPS)
        ads_connected = true;
        Serial.println("[OK] ADS1115 (0x48: MyoWare A0 + EEG A1) initialized successfully.");
    } else {
        Serial.println("[WARNING] ADS1115 not found at 0x48! Check wiring/pull-ups on GPIO21/22.");
    }

    digitalWrite(STATUS_LED_PIN, LOW);
    last_sample_time_us = micros();
    Serial.println("[INFO] 100 Hz sensor acquisition loop running...\n");
}

// ==============================================================================
// MAIN LOOP (100 Hz / 10ms cycle)
// ==============================================================================
void loop() {
    uint32_t current_time_us = micros();

    // Strictly timed 100 Hz execution loop
    if ((current_time_us - last_sample_time_us) >= TARGET_SAMPLE_PERIOD_US) {
        last_sample_time_us += TARGET_SAMPLE_PERIOD_US;

        // 1. Read 5x FSR force sensors from ESP32 ADC (GPIO32, 33, 34, 35, 36)
        for (int i = 0; i < 5; i++) {
            int raw_val = analogRead(FSR_PINS[i]);
            float v = esp32_adc_to_volts(raw_val);
            filtered_fsr[i] = apply_ema(v, filtered_fsr[i], EMA_ALPHA_FSR);
        }

        // 2. Read MyoWare EMG (A0) and EEG (A1) from ADS1115
        float v_emg = 0.0f;
        float v_eeg = 0.0f;

        if (ads_connected) {
            int16_t raw_emg = ads.readADC_SingleEnded(0); // A0: MyoWare SIG
            int16_t raw_eeg = ads.readADC_SingleEnded(1); // A1: EEG OUT
            v_emg = ads_to_volts(raw_emg);
            v_eeg = ads_to_volts(raw_eeg);
        }

        filtered_emg = apply_ema(v_emg, filtered_emg, EMA_ALPHA_EMG);
        filtered_eeg = apply_ema(v_eeg, filtered_eeg, EMA_ALPHA_EEG);

        // 3. Construct 100Hz JSON Telemetry Packet
        // Format: {"seq":123,"fsr":[0.12,0.45,0.00,0.01,0.05],"emg":0.82,"eeg":0.34,"ts":4520}
        char json_buffer[192];
        snprintf(json_buffer, sizeof(json_buffer),
                 "{\"seq\":%lu,\"fsr\":[%.3f,%.3f,%.3f,%.3f,%.3f],\"emg\":%.3f,\"eeg\":%.3f,\"ts\":%lu}\n",
                 packet_seq++,
                 filtered_fsr[0], filtered_fsr[1], filtered_fsr[2], filtered_fsr[3], filtered_fsr[4],
                 filtered_emg,
                 filtered_eeg,
                 millis());

        // 4. Stream Packet over Serial2 to Jetson Orin Nano
        Serial2.print(json_buffer);

        // 5. Diagnostic Echo to USB Serial every 50 packets (2 Hz)
        if (packet_seq % 50 == 0) {
            Serial.print("[TX] ");
            Serial.print(json_buffer);
        }
    }

    // 6. Visual Heartbeat Indicator (Blinks at 1 Hz)
    if (millis() - last_heartbeat_ms >= 500) {
        last_heartbeat_ms = millis();
        led_state = !led_state;
        digitalWrite(STATUS_LED_PIN, led_state ? HIGH : LOW);
    }
}
