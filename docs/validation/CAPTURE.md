# RealSense D435i Real-World Dataset Capture & Labeling Protocol

This protocol defines the standardized data acquisition procedure for collecting the **Human Test Dataset** for VAPA.

> [!IMPORTANT]
> **Data Integrity Invariant**: The human's captured dataset is reserved **strictly for testing and deployment validation**. It must **NEVER** be merged into training or validation splits, nor used for hyperparameter tuning.

---

## 1. Physical Hardware Setup

1. **Camera Mounting**: Mount the Intel RealSense D435i on the upper forearm mount of the VAPA robotic arm or on a calibrated tabletop tripod aligned with the base coordinate frame:
   - Base Offset: $[X=0.05\text{ m}, Y=0.15\text{ m}, Z=0.25\text{ m}]$
   - Mounting Pitch: $15.0^\circ$ downward tilt.
2. **Cable Connection**: Use a certified USB 3.1 Gen 1 Type-C cable connected directly to the NVIDIA Jetson Orin / development laptop (avoid unpowered USB hubs). Verify RealSense USB 3.2 SuperSpeed link:
   ```bash
   rs-enumerate-devices | grep -i "usb"
   ```
3. **Capture Resolution**:
   - Color Stream: $640 \times 480$ @ 30 FPS (RGB8)
   - Depth Stream: $640 \times 480$ @ 30 FPS (Z16 aligned to color)

---

## 2. Experimental Factor Matrix (Per Class Target: 200–500 Frames)

To guarantee generalized perception across clinical and everyday environments, collect images across the following test matrix:

| Factor | Required Variations | Minimum Samples / Condition |
| :--- | :--- | :---: |
| **Distance Bands** | **4 Distances**: `0.30 m`, `0.50 m`, `0.80 m`, `1.20 m` | 50 images per distance |
| **Viewing Angles** | **3 Angles**: Frontal ($0^\circ$), Oblique ($45^\circ$), Top-Down ($75^\circ$) | 40 images per angle |
| **Illumination** | **3 Lighting Modes**: Bright (~500 lux), Dim Indoor (~100 lux), Direct Sunlight / Glare | 40 images per mode |
| **Background Complexity** | **2 Modes**: Clean (monochrome desk) vs Cluttered (kitchen counter, desktop office tools) | 60 images per mode |
| **Human Hand Occlusion** | **2 Conditions**: Clear view vs Natural hand approaching/holding object | 40 images per mode |

---

## 3. Interactive Capture Command Line Execution

Execute `tools/capture_dataset.py` for each object class:

```bash
# Example: Capturing an Apple at 0.5m in bright lighting with background clutter
PYTHONPATH=. .venv/bin/python tools/capture_dataset.py \
  --class apple \
  --distance 0.5 \
  --lighting bright \
  --scene kitchen_counter_01 \
  --background cluttered

# Example: Capturing a Mug at 0.3m with human hand reaching in view
PYTHONPATH=. .venv/bin/python tools/capture_dataset.py \
  --class mug \
  --distance 0.3 \
  --lighting dim \
  --scene office_desk_02 \
  --hands \
  --burst 10
```

### Controls During Interactive Capture:
- **`[SPACE]`**: Capture and save single synchronized RGB JPEG, 16-bit Depth PNG, and JSON metadata.
- **`[B]`**: Burst mode (records 10 frames spaced by 100ms with natural camera vibration).
- **`[Q]` / `[ESC]`**: Terminate session cleanly.

---

## 4. Directory Structure of Saved Captures

Captures are stored in `data/raw/user_captures/<class>/`:
```
data/raw/user_captures/
└── mug/
    ├── rgb/
    │   └── mug_tabletop_0.5m_1728345600_0001.jpg
    ├── depth/
    │   └── mug_tabletop_0.5m_1728345600_0001_depth.png (16-bit uint PNG, millimeters)
    └── metadata/
        └── mug_tabletop_0.5m_1728345600_0001.json
```

---

## 5. Labeling Workflow (CVAT / Label Studio)

### A. Recommended Tool: CVAT (Computer Vision Annotation Tool)
1. **Launch Local CVAT via Docker**:
   ```bash
   git clone https://github.com/cvat-ai/cvat && cd cvat
   docker compose up -d
   ```
2. **Project Setup**:
   - Create project: `VAPA_Test_Set`
   - Paste the 14 standard classes from `data/classes.yaml`:
     `mug, bottle, can, apple, orange, spoon, fork, cup, banana, box, book, cell phone, mouse, remote`.
3. **Bounding Box Standards**:
   - Draw tight rectangular bounding boxes around the full visible boundary of the object.
   - For **mugs**: Enclose the mug body **and** the handle in one single bounding box.
   - For **occluded objects**: Label the object if at least $30\%$ of its surface is visible.

### B. Export & Conversion to YOLO Format
1. Export task from CVAT in **YOLO 1.1 format** (ZIP containing `.txt` files).
2. Unzip into `data/annotations/cvat_export/`.
3. Ingest annotations using the VAPA importer:
   ```bash
   PYTHONPATH=. .venv/bin/python tools/data/import_annotations.py \
     --annotations data/annotations/cvat_export/ \
     --out data/yolo_dataset/
   ```
