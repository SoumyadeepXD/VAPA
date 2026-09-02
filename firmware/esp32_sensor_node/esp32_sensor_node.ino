/*
================================================================================
 VAPA - Visually Assisted Prosthetic Arm
 NODE 2: ESP32 Bio-Signal & Tactile Sensor Acquisition Firmware
================================================================================
 Hardware Setup:
 - MCU: ESP32 NodeMCU / DevKit V1
 - I2C Bus: GPIO 21 (SDA), GPIO 22 (SCL) with 4.7k ohm pull-ups to 3.3V
 - ADC Module #1: ADS1115 (0x48, ADDR -> GND)
     * A0: Thumb FSR 402 (10k divider + 100nF cap)
     * A1: Index FSR 402 (10k divider + 100nF cap)
     * A2: Middle FSR 402 (10k divider + 100nF cap)
     * A3: Ring FSR 402 (10k divider + 100nF cap)
 - ADC Module #2: ADS1115 (0x49, ADDR -> 3.3V/VDD)
     * A0: MyoWare 2.0 EMG Sensor (Muscle envelope / raw)
     * A1: Analog EEG Sensor Module (Brainwave analog input)
     * A2: Tied to GND via 10k resistor
     * A3: Tied to GND via 10k resistor
 - Inter-Node Communication: Serial2 (GPIO 16 RX, GPIO 17 TX) @ 115200 Baud -> Jetson Orin /dev/ttyTHS1
 - Target Sampling Rate: 100 Hz (10ms loop cycle)
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

// Sampling Period (100 Hz -> 10,000 microseconds)
#define TARGET_SAMPLE_PERIOD_US 10000

// ADS1115 I2C Addresses
#define ADS1115_FSR_ADDR    0x48    // ADDR pin -> GND
#define ADS1115_BIO_ADDR    0x49    // ADDR pin -> VDD (3.3V)

// ADC Voltage Scale: GAIN_ONE gives +/- 4.096V range (1 bit = 0.125mV)
#define ADS_VOLTS_PER_BIT   0.000125f

// Digital Low-Pass / Exponential Moving Average (EMA) Alpha factors
// alpha = dt / (RC + dt). Higher alpha = faster response, lower alpha = heavier filtering.
#define EMA_ALPHA_FSR       0.35f   // Fast tactile response
#define EMA_ALPHA_EMG       0.25f   // Smooth muscle envelope
#define EMA_ALPHA_EEG       0.20f   // Filter high-frequency noise

// ==============================================================================
// GLOBAL OBJECTS & STATE
// ==============================================================================
Adafruit_ADS1115 ads_fsr;  // Module 1 (0x48)
Adafruit_ADS1115 ads_bio;  // Module 2 (0x49)

bool ads_fsr_connected = false;
bool ads_bio_connected = false;

// Filtered Signal Buffers
float filtered_fsr[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Thumb, Index, Middle, Ring
float filtered_emg    = 0.0f;
float filtered_eeg    = 0.0f;

// Frame Counter & Timing
uint32_t packet_seq = 0;
uint32_t last_sample_time_us = 0;
uint32_t last_heartbeat_ms = 0;
bool led_state = false;

// ==============================================================================
// FAST DIGITAL EXPONENTIAL MOVING AVERAGE (EMA) FILTER
// ==============================================================================
inline float apply_ema(float raw, float previous, float alpha) {
    return (alpha * raw) + ((1.0f - alpha) * previous);
}

// ==============================================================================
// FAST VOLTAGE CONVERSION HELPER
// ==============================================================================
inline float adc_to_voltage(int16_t raw_counts) {
    if (raw_counts < 0) raw_counts = 0;
    return (float)raw_counts * ADS_VOLTS_PER_BIT;
}

// ==============================================================================
// INITIALIZATION
// ==============================================================================
void setup() {
    // 1. Initialize Diagnostic USB Serial & Inter-Node High-Speed Serial2
    Serial.begin(115200);
    Serial2.begin(UART2_BAUD_RATE, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, HIGH);

    delay(100);
    Serial.println("\n========================================================");
    Serial.println(" VAPA Node 2: ESP32 Bio-Signal & Tactile Sensor Node");
    Serial.println("========================================================");

    // 2. Initialize Hardware I2C Master Bus
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_CLOCK_FREQ_HZ);

    // 3. Initialize ADS1115 #1 (FSR 402 Tactile Sensors @ 0x48)
    if (ads_fsr.begin(ADS1115_FSR_ADDR)) {
        ads_fsr.setGain(GAIN_ONE);  // +/- 4.096V range (0.125mV/LSB)
        ads_fsr.setDataRate(RATE_ADS1115_860SPS); // Maximum conversion speed (860 SPS)
        ads_fsr_connected = true;
        Serial.println("[OK] ADS1115 #1 (FSR Tactile @ 0x48) initialized.");
    } else {
        Serial.println("[ERROR] Failed to detect ADS1115 #1 at address 0x48! Check wiring.");
    }

    // 4. Initialize ADS1115 #2 (EMG & EEG Bio-Signals @ 0x49)
    if (ads_bio.begin(ADS1115_BIO_ADDR)) {
        ads_bio.setGain(GAIN_ONE);  // +/- 4.096V range
        ads_bio.setDataRate(RATE_ADS1115_860SPS);
        ads_bio_connected = true;
        Serial.println("[OK] ADS1115 #2 (Bio-Signals @ 0x49) initialized.");
    } else {
        Serial.println("[ERROR] Failed to detect ADS1115 #2 at address 0x49! Check wiring.");
    }

    digitalWrite(STATUS_LED_PIN, LOW);
    last_sample_time_us = micros();
    Serial.println("[INFO] Starting 100 Hz acquisition loop streaming to Serial2...\n");
}

// ==============================================================================
// MAIN ACQUISITION & TRANSMISSION LOOP (100 Hz)
// ==============================================================================
void loop() {
    uint32_t current_time_us = micros();

    // Enforce strictly timed 100 Hz execution loop (10ms cycle)
    if ((current_time_us - last_sample_time_us) >= TARGET_SAMPLE_PERIOD_US) {
        last_sample_time_us += TARGET_SAMPLE_PERIOD_US;

        int16_t raw_fsr[4] = {0, 0, 0, 0};
        int16_t raw_emg = 0;
        int16_t raw_eeg = 0;

        // 1. Read ADS1115 #1 (FSR 402 Channels A0 - A3)
        if (ads_fsr_connected) {
            raw_fsr[0] = ads_fsr.readADC_SingleEnded(0); // Thumb
            raw_fsr[1] = ads_fsr.readADC_SingleEnded(1); // Index
            raw_fsr[2] = ads_fsr.readADC_SingleEnded(2); // Middle
            raw_fsr[3] = ads_fsr.readADC_SingleEnded(3); // Ring
        }

        // 2. Read ADS1115 #2 (Bio-Signal Channels A0 - A1)
        if (ads_bio_connected) {
            raw_emg = ads_bio.readADC_SingleEnded(0);    // MyoWare 2.0 EMG
            raw_eeg = ads_bio.readADC_SingleEnded(1);    // EEG Brainwave Module
        }

        // 3. Convert to Volts & Apply Real-Time Low-Pass Filters
        for (int i = 0; i < 4; i++) {
            float v = adc_to_voltage(raw_fsr[i]);
            filtered_fsr[i] = apply_ema(v, filtered_fsr[i], EMA_ALPHA_FSR);
        }

        float v_emg = adc_to_voltage(raw_emg);
        float v_eeg = adc_to_voltage(raw_eeg);

        filtered_emg = apply_ema(v_emg, filtered_emg, EMA_ALPHA_EMG);
        filtered_eeg = apply_ema(v_eeg, filtered_eeg, EMA_ALPHA_EEG);

        // 4. Construct Compact High-Speed JSON Telemetry Frame
        // Format: {"seq":123,"fsr":[0.12,0.45,0.00,0.01],"emg":0.82,"eeg":0.34,"ts":4520}
        char json_buffer[160];
        snprintf(json_buffer, sizeof(json_buffer),
                 "{\"seq\":%lu,\"fsr\":[%.3f,%.3f,%.3f,%.3f],\"emg\":%.3f,\"eeg\":%.3f,\"ts\":%lu}\n",
                 packet_seq++,
                 filtered_fsr[0], filtered_fsr[1], filtered_fsr[2], filtered_fsr[3],
                 filtered_emg,
                 filtered_eeg,
                 millis());

        // 5. Transmit Packet over Serial2 to Jetson Orin
        Serial2.print(json_buffer);

        // Optional Diagnostic Echo over USB Serial every 50 frames (2 Hz)
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
