# VAPA EMG/EEG Biosignal Integration & Clinical Hardware Audit

> **Document Status**: Hardware Compatibility Audit & Integration Specification  
> **Author**: VAPA Autonomous Engineering System  
> **Date**: October 2026  
> **Target Platform**: NVIDIA Jetson Orin + ESP32 DevKit V1 + ADS1115 + MyoWare 2.0  

---

## 1. Executive Summary & Import Manifest

A machine learning pipeline and pre-recorded dataset for surface electromyography (sEMG) gesture classification was received from the team (`incoming/emg_team/`). The backup repository in `incoming/emg_team/` remains untouched.

All assets have been imported into the following structured VAPA workspace locations:
* **Raw Datasets**: `data/emg/raw/` (CSVs tracked via `data/emg/MANIFEST.csv`, large raw recordings ignored via `.gitignore`, representative sample files committed under `data/emg/raw/*_sample.csv`).
* **Training & Analysis Tools**: `tools/emg_training/` (all scripts copied byte-for-byte; visualization and plotting tools are isolated under `tools/emg_training/` rather than `biosignals/`).
* **Binary Model Artifacts**: `biosignals/models/` (`svm_emg_model.pkl`, `emg_scaler.pkl`, `feature_columns.pkl` secured with SHA-256 integrity verification).
* **Technical Documentation**: `docs/EMG.md` (this audit, reproduction report, compatibility gate, and integration architecture).

### 1.1 Cryptographic Import Manifest (`data/emg/MANIFEST.csv`)

| File Path | SHA-256 Checksum | Rows | Size (bytes) | Status |
| :--- | :--- | :--- | :--- | :--- |
| `data/emg/raw/features.csv` | `0e27ceaeed48e569087d03769da0637ce16c54def225786ca8809c36b882e24e` | 256 | 22,353 | Verified |
| `data/emg/raw/features_frequency.csv` | `8d0cb88f1011455636497cf20f51e59460be76e65ec9c638f52af1963e893ac6` | 256 | 33,343 | Verified |
| `data/emg/raw/fist.csv` | `8b3584d299acb1002f914c025c6619f88c37e514a4fe7e36aef239d009cd038a` | 15,003 | 220,981 | Verified |
| `data/emg/raw/fist_sample.csv` | `6861ab41e490fab46341282f1ab37acd616deec16aed87e285044a5dfffbd298` | 101 | 1,317 | Committed Sample |
| `data/emg/raw/open.csv` | `8a58fdae7966ee709b4093df4e704c4759694e8b94613d7cc29a06c197ad5974` | 15,002 | 220,963 | Verified |
| `data/emg/raw/open_sample.csv` | `8b1554a05a6f3591763c15d53249ebcd6dda77ad9eb8d852433f9fb701e15a63` | 101 | 1,314 | Committed Sample |
| `data/emg/raw/rest.csv` | `b850329ba5c081feb691193568e9faf5b91d3d0f8f79e2c1aac266818371125b` | 15,002 | 220,962 | Verified |
| `data/emg/raw/rest_sample.csv` | `9feadf28c116784580bed931c121789d37816bcff48ac7a02ad76c9e4e40cf1a` | 101 | 1,317 | Committed Sample |
| `data/emg/raw/test.csv` | `be09343e3ee858cc4e6c2817de908c911bf2189a756c8e53a5294b8429b9bbe1` | 15,002 | 220,966 | Verified |
| `data/emg/raw/test_sample.csv` | `efa0ca161db5c7d0bf672a4a76219bcecd7891265b1f24050ea48a1ac082bf3c` | 101 | 1,317 | Committed Sample |
| `data/emg/raw/wrist_extension.csv` | `df48c1f397c7cadf2e21deb1b702fb5e06f578ac97ca3935775ff436f28fee3e` | 15,002 | 220,946 | Verified |
| `data/emg/raw/wrist_extension_sample.csv` | `d7a9ed9e9481dca3cbba52dbccfae4154e81c5fc5c2accc2b326bd3cf6f83d1b` | 101 | 1,317 | Committed Sample |
| `data/emg/raw/wrist_flexion.csv` | `7e6fa1724c245e677580da7fe24d82b2b287c3090fb5d4a633ed7dfd4fee3e59` | 15,002 | 220,960 | Verified |
| `data/emg/raw/wrist_flexion_sample.csv` | `f2bc353bbe7361cd84feac75175d6247cdd359ff42db54cc398a3ff22bb41b90` | 101 | 1,317 | Committed Sample |
| `biosignals/models/emg_scaler.pkl` | `10af92fcd4642c048b7c9a8322025ebf5330118ce00e7be9a384974468cf3ad1` | N/A | 1,191 | Verified Scaler |
| `biosignals/models/feature_columns.pkl` | `a039bf613659a6235a21ba0a8c887dbdfbc76a4545b02bf8ede50e10ad2c0f47` | N/A | 140 | 10 Column Headers |
| `biosignals/models/svm_emg_model.pkl` | `5a1dd23d78109578f0d8bc93d04850e3f3114b3b198fc68541e70bb2208ba4ac` | N/A | 22,543 | Trained RBF SVC |

---

## 2. Teammate Model & Hardware Audit

### 2.1 Teammate Questionnaire & Provenance Matrix

| Field | Value | Provenance | Notes & Context |
| :--- | :--- | :--- | :--- |
| **Sensor / Module & Model** | Single-channel analog EMG front-end | `FROM_CODE` | Serial data streamed over COM3 @ 115200 baud (`record_emg.py`). Raw integer ADC counts (0–4095) centered at ~1905 counts. Exact front-end IC model (AD8232, MyoWare RAW pin, DFRobot) is `UNKNOWN`. |
| **Channel Count** | 1 Channel | `FROM_CODE` | Serial parser accepts only `[timestamp_us, emg]`. |
| **Electrode Placement** | Single site on forearm | `UNKNOWN` | No muscle anatomical landmarks (FDS, EDC, FCR) documented. |
| **Sampling Rate ($f_s$)** | 500 Hz | `FROM_CODE` | $\Delta t = 2000\,\mu\text{s}$ ($\pm 0\,\mu\text{s}$ jitter on rest/fist/flex/ext); explicitly defined as `FS = 500`. |
| **Window Length** | 500 samples (1000 ms) | `FROM_CODE` | `WINDOW_SIZE = 500` in feature extractors. |
| **Window Stride / Overlap** | 250 samples (500 ms / 50% overlap) | `FROM_CODE` | `STEP_SIZE = 250` in feature extractors. |
| **Subject Count** | 1 Subject (`SUBJ_01`) | `FROM_CODE` | Single volunteer recording. |
| **Session Count & Days** | 1 Session, Single Day | `FROM_CODE` | All recordings created on August 14 between 00:40 and 01:26. No multi-day data exist. |
| **Reported Accuracies** | 70.51% – 76.92% | `FROM_CODE` | RF Time (70/30 split): 70.51%; RF Time+Freq (70/30 split): 74.36%; SVM Time+Freq (70/30 split): 76.92%. |
| **Evaluation Method** | 70/30 Temporal Split / 80/20 Random | `FROM_CODE` | Split within each continuous 30-second recording. |
| **Pinned Library Versions** | Python 3.12.3, scikit-learn 1.8.0 | `FROM_CODE` | Runtime environment: `python 3.12.3`, `scikit-learn 1.8.0`, `numpy 2.4.6`, `scipy 1.17.1`, `pandas 3.0.3`, `joblib 1.5.3`. |
| **EEG Data Included?** | None included | `FROM_CODE` | No EEG dataset or file present in `incoming/emg_team/`. EEG module model: `UNKNOWN`. |

### 2.2 Signal Nature: Raw Bi-Phasic vs Envelope
* **Verdict**: The signal is **Raw Bi-Phasic sEMG**, **NOT an envelope**.
* **Spectral Evidence**: Fast Fourier Transform (FFT) reveals prominent 50 Hz powerline mains hum (spectral peak magnitude > 300,000) and 100 Hz second harmonic, along with continuous spectral density spanning 10 Hz to 220 Hz.
* **Morphological Evidence**: The signal is centered at an ADC baseline of ~1905 counts and exhibits positive and negative deflections across the mean. Zero-crossing density averages ~144 crossings per second. A rectified envelope signal would be strictly positive ($V \ge 0$), unipolar, and band-limited below 10–20 Hz with zero crossings near zero.

---

## 3. Training & Evaluation Reproduction

### 3.1 Data Leakage Analysis
The team's original scripts contained two leakage mechanisms:
1. **Sliding-Window Leakage in Random Split (`train_model.py`)**:
   Sliding windows overlap by 50% (250 shared samples between adjacent windows). In a random train/test split, adjacent windows are split across train and test partitions, allowing the model to memorize overlapping segments.
2. **Boundary-Window Overlap in Chronological Split (`train_svm.py`, `evaluate_temporal.py`)**:
   The chronological 70/30 split partitions the first 70% of each 30s recording into train and the last 30% into test. Exactly one window at the 70% boundary shares 250 samples with the previous window.
3. **Session/Electrode Leakage**:
   Because only ONE continuous 30-second recording was made per gesture, the test set represents the exact same electrode placement, skin impedance, and muscle fatigue state recorded seconds later. True inter-session generalization is unmeasured.

### 3.2 Honest Reproduction Benchmark Results

All evaluations were executed with pinned random seed (`random_state=42`) using the exact dataset and features:

| Model Architecture | Feature Set | Split Type | Accuracy | Macro F1 | Weighted F1 | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Random Forest (200)** | Time-Domain (6 feats) | Random 80/20 | 70.59% | 0.70 | 0.70 | Severe sliding-window leakage |
| **Random Forest (200)** | Time+Freq (10 feats) | Random 80/20 | 70.59% | 0.69 | 0.70 | Severe sliding-window leakage |
| **SVC (RBF, C=10)** | Time+Freq (10 feats) | Random 80/20 | 80.39% | 0.81 | 0.80 | Severe sliding-window leakage |
| **Random Forest (200)** | Time-Domain (6 feats) | Chrono 70/30 | 70.51% | 0.68 | 0.69 | Original team temporal baseline |
| **Random Forest (200)** | Time+Freq (10 feats) | Chrono 70/30 | 74.36% | 0.72 | 0.73 | Original team temporal baseline |
| **SVC (RBF, C=10)** | Time-Domain (6 feats) | Chrono 70/30 | 73.08% | 0.72 | 0.73 | Reduced time-domain SVM |
| **SVC (RBF, C=10)** | Time+Freq (10 feats) | Chrono 70/30 | 76.92% | 0.76 | 0.76 | **Teammate primary model** |
| **Random Forest (200)** | Time-Domain (6 feats) | Purged 70/30 | 71.23% | 0.69 | 0.70 | Leak-free boundary buffer |
| **Random Forest (200)** | Time+Freq (10 feats) | Purged 70/30 | 75.34% | 0.74 | 0.75 | Leak-free boundary buffer |
| **SVC (RBF, C=10)** | Time-Domain (6 feats) | Purged 70/30 | 75.34% | 0.75 | 0.75 | Leak-free boundary buffer |
| **SVC (RBF, C=10)** | Time+Freq (10 feats) | Purged 70/30 | **78.08%** | **0.78** | **0.78** | **Leak-free boundary buffer** |

### 3.3 Detailed Classification Reports & Confusion Matrices

#### A. Teammate Model (Time + Frequency SVM, Chronological 70/30 with Purged Boundary, Test N=73)
* **Overall Accuracy**: 78.08%
* **Per-Class Metrics**:
  * `FIST`: Precision = 0.91, Recall = 0.71, F1 = 0.80 (Support = 14)
  * `OPEN`: Precision = 0.80, Recall = 0.86, F1 = 0.83 (Support = 14)
  * `REST`: Precision = 0.88, Recall = 0.88, F1 = 0.88 (Support = 17)
  * `WRIST_EXTENSION`: Precision = 0.73, Recall = 0.57, F1 = 0.64 (Support = 14)
  * `WRIST_FLEXION`: Precision = 0.63, Recall = 0.86, F1 = 0.73 (Support = 14)
* **Confusion Matrix** (Rows: True, Cols: Pred; Order: FIST, OPEN, REST, EXT, FLEX):
  ```
  [[10,  1,  0,  2,  1],   <- FIST
   [ 0, 12,  0,  0,  2],   <- OPEN
   [ 1,  0, 15,  1,  0],   <- REST
   [ 0,  0,  2,  8,  4],   <- WRIST_EXTENSION (Confused with FLEXION & REST)
   [ 0,  2,  0,  0, 12]]   <- WRIST_FLEXION
  ```

#### B. Time-Domain Model (Random Forest, Chronological 70/30 with Purged Boundary, Test N=73)
* **Overall Accuracy**: 71.23%
* **Per-Class Metrics**:
  * `FIST`: Precision = 0.73, Recall = 0.79, F1 = 0.76 (Support = 14)
  * `OPEN`: Precision = 0.93, Recall = 0.93, F1 = 0.93 (Support = 14)
  * `REST`: Precision = 0.88, Recall = 0.82, F1 = 0.85 (Support = 17)
  * `WRIST_EXTENSION`: Precision = 0.33, Recall = 0.21, F1 = 0.26 (Support = 14)
  * `WRIST_FLEXION`: Precision = 0.58, Recall = 0.79, F1 = 0.67 (Support = 14)
* **Confusion Matrix**:
  ```
  [[11,  0,  0,  2,  1],   <- FIST
   [ 0, 13,  0,  0,  1],   <- OPEN
   [ 0,  0, 14,  3,  0],   <- REST
   [ 3,  0,  2,  3,  6],   <- WRIST_EXTENSION (Severe failure: 21% recall)
   [ 1,  1,  0,  1, 11]]   <- WRIST_FLEXION
  ```

> [!NOTE]
> Single-channel EMG suffers from severe anatomical crosstalk between `WRIST_EXTENSION` and `WRIST_FLEXION`. Antagonist muscle activity cannot be reliably isolated with a single bipolar electrode pair without spatial differentiation.

---

## 4. Hardware Compatibility Gate (Decision Point)

### 4.1 Specification Comparison: Training vs Deployed Pipeline

| Dimension | Training Data / Teammate Model | Live Deployed VAPA Path (`config/hardware_config.py` & ESP32) | Hardware Compatibility Verdict |
| :--- | :--- | :--- | :--- |
| **Channels** | 1 Channel | 1 Channel (ADS1115 A0) | **Channel count matches**, but 1 channel cannot isolate multi-DOF gestures without crosstalk. |
| **Sampling Rate ($f_s$)** | 500 Hz ($\Delta t = 2000\,\mu\text{s}$) | 100 Hz stream (`TARGET_SAMPLE_PERIOD_US = 10000`) | **INCOMPATIBLE**: 5x downsampling; deployed Nyquist limit is 50 Hz. |
| **Signal Type** | Raw Bi-Phasic AC EMG (ADC counts fluctuating around ~1905) | Rectified & Integrated Envelope (ENV) with EMA filter ($\alpha=0.25$, voltage $0.0-3.3\text{V}$) | **FATAL INCOMPATIBILITY**: AC bi-phasic vs unipolar low-pass smoothed envelope. |
| **Window Length** | 1000 ms (500 samples @ 500 Hz) | 50–200 ms (5–20 samples @ 100 Hz) for reactive control | **INCOMPATIBLE**: 1000 ms introduces unacceptable 1-second physical grasping lag. |
| **Electrode Placement** | Single unknown forearm site | Single flexor site (FDS / flexor compartment) | **INCOMPATIBLE / UNKNOWN**: Cannot classify antagonist (extensor) or wrist deviation without antagonist sensor. |
| **Input Scaling** | Raw ADC integer counts ($0-4095$), scaler expects $\mu_{\text{RMS}} \approx 75.0$ | Calibrated Volts ($0.0-3.3\text{V}$) or normalized activation ($0.0-1.0$) | **FATAL INCOMPATIBILITY**: Passing $0.0-3.3\text{V}$ into scaler expecting $\sim 2000$ counts yields $Z \approx -25.0$. |

### 4.2 Feature Feasibility on Deployed 100 Hz Envelope Signal

1. **`RMS` & `MAV`**: **Feasible**. Measures average envelope voltage and contraction intensity.
2. **`Variance` & `STD`**: **Feasible**. Measures envelope amplitude ripple and instability.
3. **`Energy` & `Peak_to_Peak`**: **Feasible**. Measures integrated contraction effort and peak excursion.
4. **`Waveform_Length`**: **Severely Degraded**. On raw 500 Hz EMG, this measures high-frequency MUAP interference patterns. On a 100 Hz EMA-smoothed envelope ($\alpha=0.25$), it only reflects slow macro-motion slopes.
5. **`Zero_Crossings`**: **INCOMPATIBLE**. Envelope signals are strictly non-negative ($V \ge 0$). Subtraction of window mean creates arbitrary baseline crossings driven by low-frequency drift or 1–5 Hz tremor, completely decoupling from MUAP firing rates.
6. **`Mean_Frequency` (MNF) & `Median_Frequency` (MDF)**: **INCOMPATIBLE**. Frequency features strictly require raw AC EMG sampled at $\ge 1000\text{ Hz}$ to capture the physiological 20–450 Hz spectrum. At 100 Hz, the Nyquist limit is 50 Hz, and the ESP32 EMA filter suppresses components above ~10 Hz. FFT over 100 Hz envelope yields meaningless low-frequency noise.

---

### 4.3 DECISION GATE: Compatibility Verdict

> [!CAUTION]
> **COMPATIBILITY VERDICT: INCOMPATIBLE (FATAL)**  
> The teammate's pre-trained model (`svm_emg_model.pkl`) and scaler (`emg_scaler.pkl`) **CANNOT BE DEPLOYED DIRECTLY** on the current VAPA physical hardware path.
> Directly feeding the live 100 Hz smoothed envelope voltage into this model will cause severe feature corruption, incorrect classification, 1000 ms command latency, and erratic arm behavior.

### 4.4 Options & Effort Estimates for the Human Operator

| Option | Description | Effort Estimate | Safety & System Impact |
| :--- | :--- | :--- | :--- |
| **(a) Keep Threshold Decoder (Offline Research Only)** | Keep `biosignals/emg_decoder.py` (threshold envelope decoder with proportional force) active. Retain the teammate's dataset and scripts strictly for offline research and documentation. Do not modify real-time control. | **Low** (~1–2 hours) | **Zero Risk**: Production FSM, IK, and tactile control remain 100% verified and operational. |
| **(b) Retrain Reduced Model on Deployed Features** | Engineer an envelope-compatible feature set (MAV, RMS, Variance, Peak-to-Peak, rate of change) over short $\le 200\text{ ms}$ windows. Implement `biosignals/emg_classifier.py` with safe loader, fail-closed fallback to threshold decoder, majority voting, confidence thresholding, and parallel threshold-based co-contraction E-stop invariants. Add `tools/emg_training/record_from_esp32.py` for recording matched live data. | **Medium** (~3–5 hours) | **Safe Controlled Upgrade**: Machine learning inference enabled while maintaining fail-closed safety and parallel E-stop. |
| **(c) Upgrade Hardware to Multi-Channel High-Speed Raw EMG** | Rewire physical hardware to tap MyoWare RAW output pins (or integrate an ADS1299 / multi-channel ADC front-end), upgrade ESP32 firmware to 1000 Hz binary streaming (COBS/slip UART), and position 4–8 electrodes across forearm muscle compartments. | **High** (~2–3 days hardware + firmware refactoring) | **High Hardware Effort**: Requires bench soldering, hardware rewiring, protocol redesign, and extensive hardware re-certification. |

---
