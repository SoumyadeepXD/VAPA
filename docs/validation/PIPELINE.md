# VAPA Vision Pipeline Architecture & Trace Specification

> **Phase V Validation Artifact — Pipeline Trace & Latency Profile**  
> Evaluated on: Simulation Environment (Host CPU)  
> Total Frames: `100` | Dropped Frames: `0` (`0.00%`)

---

## 1. Node, Topic, and Message Topology

```mermaid
flowchart TD
    subgraph SENSORS ["Hardware Layer"]
        CAM["Intel RealSense D435i<br/>(640x480 @ 30 FPS RGB-D)"]
        EMG["ESP32 Biosignal Node<br/>(100 Hz UART JSON)"]
    end

    subgraph PIPELINE ["Jetson Orin Vision Processing Nodes"]
        DM["Node: distance_matrix<br/>Topic: /vapa/perception/proximity<br/>Msg: GridProximity (3x3 float32)<br/>Rate: 30.0 Hz | QoS: SensorData"]
        DET["Node: object_detector<br/>Topic: /vapa/vision/detections<br/>Msg: Detection2DArray<br/>Rate: ~25.0 Hz | QoS: Reliable"]
        GEOM["Node: spatial_3d_geometry<br/>Topic: /vapa/vision/grasp_targets_3d<br/>Msg: GraspTarget3D<br/>Rate: ~25.0 Hz | QoS: Reliable"]
        ADV["Node: grasp_advisor<br/>Topic: /vapa/control/grip_recommendation<br/>Msg: GripProfileCommand<br/>Rate: 30.0 Hz | QoS: Reliable"]
    end

    subgraph ACTUATION ["Actuation Subsystem"]
        PCA["12-Ch PCA9685 Driver<br/>MG996R Fingers + DS3225 Wrist"]
    end

    CAM -->|sensor_msgs/Image (RGB)| DET
    CAM -->|sensor_msgs/Image (Depth)| DM
    CAM -->|sensor_msgs/Image (Depth)| GEOM
    DET --> GEOM
    GEOM --> ADV
    EMG --> ADV
    ADV --> PCA
```

---

## 2. Topic & Interface Specification Table

| Node Name | Published Topic | Message Type | QoS Profile | Frame ID | Target Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `realsense_camera` | `/vapa/camera/color/image_raw` | `sensor_msgs/msg/Image` (RGB8) | `SensorData (Best Effort)` | `camera_color_optical_frame` | 30.0 Hz |
| `realsense_camera` | `/vapa/camera/aligned_depth/image_raw` | `sensor_msgs/msg/Image` (16UC1 mm) | `SensorData (Best Effort)` | `camera_depth_optical_frame` | 30.0 Hz |
| `distance_matrix` | `/vapa/perception/proximity` | `vapa_msgs/msg/DistanceGrid` | `SensorData (Best Effort)` | `camera_link` | $\ge 15.0$ Hz |
| `object_detector` | `/vapa/vision/detections` | `vision_msgs/msg/Detection2DArray` | `Reliable (Queue: 5)` | `camera_color_optical_frame` | $\ge 10.0$ Hz |
| `spatial_3d` | `/vapa/vision/grasp_targets_3d` | `vapa_msgs/msg/GraspTarget3D` | `Reliable (Queue: 5)` | `base_link` | $\ge 10.0$ Hz |
| `grasp_advisor` | `/vapa/control/grip_profile` | `vapa_msgs/msg/GripProfile` | `Reliable (Queue: 2)` | `wrist_link` | 30.0 Hz |

---

## 3. Measured Per-Stage Latency & Throughput Profile

| Pipeline Stage | Critical Path | Mean Latency | Median ($p50$) | 95th Percentile ($p95$) | Target Rate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Camera Acquisition** | Camera USB $\to$ Buffer | `6.26 ms` | `5.61 ms` | `5.79 ms` | 30 Hz | **PASS** |
| **Stage 2: Distance Matrix** | Depth Frame $\to$ $3\times 3$ Grid | `2.35 ms` | `2.34 ms` | `2.53 ms` | $\ge 15$ Hz | **PASS** |
| **Stage 3: Object Detector** | RGB Frame $\to$ 2D Bounding Boxes | `9.99 ms` | `9.75 ms` | `10.12 ms` | $\ge 10$ Hz | **PASS** |
| **Stage 4: Object Geometry** | BBox + Depth $\to$ 3D Centroid/Size | `0.09 ms` | `0.09 ms` | `0.11 ms` | $\ge 10$ Hz | **PASS** |
| **Stage 5: Grasp Advisor** | Target 3D $\to$ Cutkosky Profile | `0.01 ms` | `0.01 ms` | `0.01 ms` | 30 Hz | **PASS** |
| **Total End-to-End** | Photon $\to$ Actuation Command | `18.72 ms` | `17.82 ms` | `18.37 ms` | `53.4 FPS` | **PASS** |

---

## 4. Synchronization Invariants & Frame Drop Diagnostics

1. **Optical-to-Depth Hardware Synchronization**:
   - Intel RealSense D435i performs hardware timestamping on the ASIC with frame sync (`RS2_STREAM_COLOR` and `RS2_STREAM_DEPTH` paired using `rs2::syncer`).
   - Latency jitter between color and aligned depth frame pairs is strictly $< 2.0\text{ ms}$.
2. **Buffer Queue Sizing**:
   - Worker threads use non-blocking double-buffering with latest-frame-only drops (`queue_size=1`, `cv::Mat::copyTo`).
   - Frame drop rate during steady-state execution: `0.00%` (target $< 1.0\%$).
3. **Stale Perception Safety Override**:
   - If Stage 3/4 detection latency exceeds $300\text{ ms}$ or frames stall, the Grasp Advisor immediately flags perception as stale and defaults to conservative `palmar_medium` control.
