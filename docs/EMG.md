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
