# Software & Dataset Licensing Registry

This document records the licenses and intellectual property compliance terms for all detection frameworks, weights, and datasets evaluated in **VAPA Phase V (Vision Data, Training & Validation)**.

---

## 1. Object Detection Framework & Neural Architecture

| Component | Repository / Provider | Version | License | Commercial Use Permitted? | Redistribution / Copyleft Requirement |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Ultralytics YOLO (YOLOv8 / YOLO11)** | [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) | `ultralytics >= 8.3.0` | **GNU Affero General Public License v3.0 (AGPL-3.0)** | Yes (with source disclosure) or via Commercial Enterprise License | Strong copyleft: Source code modifications and network services interacting with modified versions must remain open source under AGPL-3.0 unless enterprise commercial license is procured. |
| **ONNX Runtime** | Microsoft | `1.17+` | **MIT License** | Yes | Permissive open source license. |
| **TensorRT / PyCUDA** | NVIDIA Corporation | `TensorRT 8.6+ / 10.x` | **NVIDIA Software License Agreement** | Yes | Free for deployment on NVIDIA hardware (Jetson Orin). |

---

## 2. Public Datasets

| Dataset | Source / Registry | License | Attribution Requirement | Usage in VAPA |
| :--- | :--- | :--- | :--- | :--- |
| **COCO (Common Objects in Context)** | [cocodataset.org](https://cocodataset.org/) | **Creative Commons Attribution 4.0 (CC BY 4.0)** | Yes (Lin et al., Microsoft COCO) | Train / Val subsets for standard classes: `bottle`, `cup`, `fork`, `spoon`, `banana`, `apple`, `orange`, `cell phone`, `book`, `mouse`, `remote`. |
| **Open Images Dataset V7** | Google LLC | **Annotations: CC BY 4.0**<br>**Images: CC BY 2.0 / CC BY 4.0** | Yes (Kuznetsova et al., Google) | Class-specific subsets via FiftyOne for non-COCO classes: `Tin can`, `Box / Cardboard box`, `Mug`. |
| **Zenodo Fruit Dataset** | [Zenodo DOI 10.5281/zenodo.18618629](https://doi.org/10.5281/zenodo.18618629) | **Creative Commons Attribution 4.0 International (CC BY 4.0)** | Yes (Zenodo record) | Fine-tuning domain coverage for spherical/cylindrical fruit varieties under varied indoor lighting. |
| **ClearPose (Optional)** | Chen et al. | **Research Only (Non-Commercial)** | Yes | Transparent/glass objects (clear cups, glassware) depth validation. Excluded from commercial deployment manifests. |
| **Washington RGB-D Dataset (Optional)** | University of Washington | **CC BY 4.0 / Academic Non-Commercial** | Yes (Lai et al.) | Ground-truth size and caliper metric validation. |
| **GMU Kitchens (Optional)** | George Mason University | **Academic Non-Commercial** | Yes (Georgakis et al.) | Cluttered countertop multi-object evaluation. |

---

## 3. Human's Own Capture Dataset (Test Set)

| Dataset | Source | License | Usage |
| :--- | :--- | :--- | :--- |
| **VAPA Hand-Mounted D435i Test Captures** | VAPA Project Operator | **Proprietary / Internal VAPA Project License** | Held out **EXCLUSIVELY** for final testing and benchmark evaluation. **Zero leakage into training or validation splits.** |
