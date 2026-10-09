# VAPA EMG Dataset & Teammate Model Audit

> **Dataset Location**: `data/emg/raw/`  
> **Manifest**: `data/emg/MANIFEST.csv`  
> **Model Artifacts**: `biosignals/models/`  
> **Training Scripts**: `tools/emg_training/`  
> **Audit Status**: Complete Codebase & Signal Inspection (Step 2)

---

## 1. Teammate Questionnaire & Signal Provenance

All fields are verified and tagged with provenance: `FROM_CODE` (extracted directly from code or data files), `FROM_TEAMMATE` (explicitly stated in teammate notes), or `UNKNOWN` (missing from code, data, and documentation).

| Field | Value | Provenance | Notes |
| :--- | :--- | :--- | :--- |
| **Sensor / Module & Model** | Single-channel analog EMG front-end | `FROM_CODE` | Serial data streamed over COM3 @ 115200 baud (`record_emg.py`). Outputs 12-bit ADC integers (0–4095) centered at ~1905 counts. Exact manufacturer/IC (e.g., AD8232, MyoWare RAW pin, DFRobot) is `UNKNOWN`. |
| **Channel Count** | 1 Channel | `FROM_CODE` | `record_emg.py` accepts strictly 2 comma-separated fields: `timestamp_us, emg`. |
| **Electrode Placement** | Single site on forearm | `UNKNOWN` | No muscle anatomical landmark (e.g., FDS, EDC, FCR) is specified in code, headers, or comments. |
| **Sampling Rate ($f_s$)** | 500 Hz | `FROM_CODE` | Timestamps show $\Delta t = 2000\,\mu\text{s}$ ($\pm 0\,\mu\text{s}$ jitter on rest/fist/flex/ext); explicitly defined as `FS = 500` in feature extractors. |
| **Window Length** | 500 samples (1000 ms) | `FROM_CODE` | `WINDOW_SIZE = 500` in `extract_features.py` and `frequency_features.py`. |
| **Window Stride / Overlap** | 250 samples (500 ms / 50% overlap) | `FROM_CODE` | `STEP_SIZE = 250` in feature extractors. |
| **Subject Count** | 1 Subject (`SUBJ_01`) | `FROM_CODE` | Single participant protocol. |
| **Session Count & Days** | 1 Session, Single Day | `FROM_CODE` | All recordings created on August 14 between 00:40 and 01:26. No multi-day or multi-session data exist. |
| **Reported Accuracies** | 70.51% – 76.92% | `FROM_CODE` | RF Time (70/30 split): 70.51%; RF Time+Freq (70/30 split): 74.36%; SVM Time+Freq (70/30 split): 76.92%. Full model (`save_model.py`) trained on 100% of data. |
| **Evaluation Method** | 70/30 Temporal Split / 80/20 Random | `FROM_CODE` | Tested in `train_model.py` (80/20 random stratified) and `evaluate_temporal.py` / `train_svm.py` (first 70% train, last 30% test). |
| **Python & Scikit-Learn Versions** | Python 3.12.3, scikit-learn 1.8.0 | `FROM_CODE` / `UNKNOWN` | Environment runs Python 3.12.3 and scikit-learn 1.8.0. Serialized pickles lack embedded `_sklearn_version` attribute. |
| **EEG Data Included?** | None included | `FROM_CODE` | No EEG CSV or stream found in `incoming/emg_team/`. EEG module model: `UNKNOWN`. |

---

## 2. Signal Characterization & Value Distribution

Analysis of all 6 raw recording CSVs in `data/emg/raw/` (15,001–15,002 samples each, 30.00 seconds duration):

| File Name | Assigned Label | Duration (s) | Sample Count | EMG Min | EMG Max | EMG Mean | EMG Std | Baseline Offset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rest.csv` | `REST` | 30.00 | 15,001 | 659 | 2,126 | 1,905.85 | 72.10 | ~1.53 V (Mid-Rail) |
| `fist.csv` | `FIST` | 30.00 | 15,002 | 1,385 | 3,150 | 1,907.15 | 107.66 | ~1.53 V |
| `open.csv` | `OPEN` | 31.86 | 15,001 | 1,535 | 2,154 | 1,869.96 | 70.15 | ~1.50 V |
| `wrist_flexion.csv` | `WRIST_FLEXION` | 30.00 | 15,001 | 534 | 2,212 | 1,908.55 | 80.52 | ~1.53 V |
| `wrist_extension.csv` | `WRIST_EXTENSION` | 30.00 | 15,001 | 0 | 2,480 | 1,899.06 | 113.54 | ~1.52 V |
| `test.csv` | *Unlabeled (Trial)* | 30.00 | 15,001 | 1,323 | 4,095 | 1,913.16 | 205.10 | Saturated at 4095 |

### 2.1 Signal Nature: Raw Bi-Phasic vs Envelope
* **Verdict**: The signal is **Raw Bi-Phasic sEMG**, **NOT an envelope**.
* **Spectral Evidence**: Fast Fourier Transform (FFT) reveals prominent 50 Hz powerline mains hum (amplitude peak > 300,000) and 100 Hz second harmonic, along with continuous spectral density spanning 10 Hz to 220 Hz.
* **Morphological Evidence**: The signal is centered at an ADC baseline of ~1905 counts and exhibits positive and negative deflections across the mean. Zero-crossing density averages ~144 crossings per second. A rectified envelope signal would be strictly positive ($V \ge 0$), unipolar, and band-limited below 10–20 Hz with zero crossings near zero.

---

## 3. Feature Extraction & Engineering Pipeline

The pipeline implements two feature extractors (`tools/emg_training/extract_features.py` and `tools/emg_training/frequency_features.py`):

### 3.1 Preprocessing & Windowing
* **Baseline Correction**: DC offset subtraction per window: $x_{\text{norm}}[n] = x[n] - \mu_x$.
* **Digital Filtering**: **None**. No bandpass filter (20–450 Hz) and no notch filter (50/60 Hz) are applied. The 50 Hz mains hum is unfiltered in both feature sets.
* **Relaxation Trimming**: For active gesture files (`fist.csv`, `open.csv`, `wrist_flexion.csv`, `wrist_extension.csv`), the first 5 seconds (2500 samples) are discarded because the protocol required resting for 5 seconds. `rest.csv` does not discard the first 5 seconds.
* **Window Size**: 500 samples (1000 ms).
* **Step Size**: 250 samples (500 ms, 50% overlap).
* **Sample Count**:
  * `REST`: 59 windows
  * `FIST`: 49 windows
  * `OPEN`: 49 windows
  * `WRIST_FLEXION`: 49 windows
  * `WRIST_EXTENSION`: 49 windows
  * **Total Samples**: 255 windows

### 3.2 Feature Definitions

| Feature Index | Feature Name | Domain | Mathematical Formulation | Live Deployed Feasibility |
| :---: | :--- | :--- | :--- | :---: |
| 1 | `RMS` | Time | $\sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}$ | Feasible |
| 2 | `MAV` | Time | $\frac{1}{N}\sum_{i=1}^N \|x_i\|$ | Feasible |
| 3 | `Variance` | Time | $\frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2$ | Feasible |
| 4 | `STD` | Time | $\sqrt{\text{Variance}}$ | Feasible |
| 5 | `Waveform_Length` | Time | $\sum_{i=2}^N \|x_i - x_{i-1}\|$ | Limited (Envelope slope) |
| 6 | `Zero_Crossings` | Time | $\sum \mathbf{1}_{(x_{i-1}x_i < 0 \land \|x_{i-1}-x_i\| \ge 0.01\sigma)}$ | **Incompatible** on envelope |
| 7 | `Mean_Frequency` | Frequency | $\frac{\sum f_k P_k}{\sum P_k}$ from FFT | **Incompatible** on 100 Hz envelope |
| 8 | `Median_Frequency` | Frequency | $f_{1/2}: \sum_0^{f_{1/2}} P_k = \frac{1}{2}\sum P_k$ | **Incompatible** on 100 Hz envelope |
| 9 | `Energy` | Time/Energy | $\sum_{i=1}^N x_i^2$ | Feasible |
| 10 | `Peak_to_Peak` | Amplitude | $\max(x) - \min(x)$ | Feasible |

---

## 4. Models & Evaluation Scripts

1. `train_model.py`: Random Forest (200 trees) on 6 time-domain features using 80/20 stratified random split. **Severe data leakage** due to overlapping sliding windows in both train and test partitions.
2. `evaluate_temporal.py`: Random Forest (200 trees) on 6 time-domain features using 70/30 chronological split.
3. `train_frequency_model.py`: Random Forest (200 trees) on 10 time+frequency features using 70/30 chronological split.
4. `train_svm.py`: Support Vector Classifier (RBF kernel, $C=10$, $\gamma=\text{scale}$) on 10 standardized features using 70/30 chronological split.
5. `save_model.py`: Retrains the RBF SVC on 100% of `features_frequency.csv` with `probability=True`, serializing `svm_emg_model.pkl`, `emg_scaler.pkl`, and `feature_columns.pkl`.
