"""
VAPA Phase 1: 3D Environmental Perception & Spatial Localization Suite
Executes complete 3D visual perception pipeline:
1. RealSense RGB-D camera acquisition & intrinsics calibration
2. Semantic object detection & 2D bounding boxes
3. Robust statistical depth extraction & 3D pinhole deprojection
4. Homogeneous Camera-to-Robot-Base coordinate transformation (T_base_cam)
5. Metric object bounding box (width, height, depth) & grasp pose orientation
6. Adaptive grasp aperture & class-specific force computation
7. Reachability verification against robot arm workspace envelope
8. Visual HUD rendering & snapshot generation
"""

import sys
import os
import time
import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vision.realsense_camera import RealSenseCamera, SyntheticRealSenseCamera
from vision.object_detector import ObjectDetector
from vision.spatial_3d import Spatial3DAnalyzer, GraspTarget3D
from vision.visualizer_3d import VisionVisualizer
from config.system_config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    MIN_VALID_DEPTH_M,
    MAX_VALID_DEPTH_M,
    WORKSPACE_BOUNDS_M,
    OBJECT_FORCE_MAP_N,
)


class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str):
    print("\n" + "=" * 78)
    print(f" {Color.BOLD}{Color.CYAN}PHASE 1 PERCEPTION: {title}{Color.RESET}")
    print("=" * 78)


def print_check(name: str, passed: bool, details: str = ""):
    status = f"{Color.GREEN}[PASSED]{Color.RESET}" if passed else f"{Color.RED}[FAILED]{Color.RESET}"
    detail_str = f" - {details}" if details else ""
    print(f"  {status} {name}{detail_str}")
    return passed


def run_phase_1_perception(force_mock: bool = True, output_snapshot: str = "phase1_perception_snapshot.png") -> bool:
    start_time = time.time()
    total_checks = 0
    passed_checks = 0

    print("\n" + "#" * 78)
    print(f"#{Color.BOLD}{Color.GREEN}        VAPA (VISUALLY ASSISTED PROSTHETIC ARM) — PHASE 1 PERCEPTION        {Color.RESET}#")
    print("#" * 78)

    # --------------------------------------------------------------------------
    # Step 1: Initialize Camera & Extract Intrinsics
    # --------------------------------------------------------------------------
    print_header("1. RGB-D Camera Sensor Initialization & Intrinsics")
    cam = RealSenseCamera(force_mock=force_mock)
    intrinsics = cam.get_intrinsics_dict()
    
    cam_init_ok = intrinsics["width"] == CAMERA_WIDTH and intrinsics["height"] == CAMERA_HEIGHT
    total_checks += 1
    if print_check("Camera Sensor Stream Initialization", cam_init_ok, f"{intrinsics['width']}x{intrinsics['height']} @ {CAMERA_FPS}Hz"):
        passed_checks += 1

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    focal_ok = (fx > 0) and (fy > 0) and (cx > 0) and (cy > 0)
    total_checks += 1
    if print_check("Camera Pinhole Model Intrinsics", focal_ok, f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"):
        passed_checks += 1

    # Grab Test Frame
    color, depth = cam.get_frames()
    frame_ok = (color is not None) and (depth is not None) and (depth.dtype in (np.float32, np.float64))
    total_checks += 1
    if print_check("RGB-D Synchronized Frame Grab", frame_ok, f"Color: {color.shape}, Depth metric: {depth.shape}"):
        passed_checks += 1

    # Check Depth Metric Range
    valid_depth_mask = (depth >= MIN_VALID_DEPTH_M) & (depth <= MAX_VALID_DEPTH_M)
    valid_ratio = float(np.count_nonzero(valid_depth_mask)) / float(depth.size)
    depth_range_ok = valid_ratio > 0.30  # At least 30% of field of view contains valid workspace depth
    total_checks += 1
    if print_check("Metric Depth Range Validation", depth_range_ok, f"{valid_ratio*100:.1f}% pixels in [{MIN_VALID_DEPTH_M}m, {MAX_VALID_DEPTH_M}m]"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 2: 3D Object Detection & Semantic Classification
    # --------------------------------------------------------------------------
    print_header("2. 3D Object Detection & Semantic Classification")
    detector = ObjectDetector()
    detections = detector.detect(color, depth_image_m=depth)
    
    detect_ok = len(detections) >= 1
    total_checks += 1
    if print_check("Object Detection Inference", detect_ok, f"Detected {len(detections)} candidate object(s)"):
        passed_checks += 1

    for idx, d in enumerate(detections):
        x0, y0, x1, y1 = d.as_int_box()
        print(f"    Candidate #{idx+1}: '{d.label.upper()}' (Confidence: {d.score*100:.1f}%) | BBox: [{x0}, {y0}] -> [{x1}, {y1}]")

    # --------------------------------------------------------------------------
    # Step 3: Spatial 3D Scene Analysis & Metric Localization
    # --------------------------------------------------------------------------
    print_header("3. Spatial 3D Scene Analysis & Metric Localization")
    spatial = Spatial3DAnalyzer()
    targets = spatial.process_scene(detections, color, depth, cam)

    targets_ok = len(targets) >= 1
    total_checks += 1
    if print_check("3D Metric Deprojection & Base Transform", targets_ok, f"Resolved {len(targets)} 3D grasp target(s)"):
        passed_checks += 1

    print("\n  Resolved 3D Grasp Targets:")
    reachable_count = 0
    for idx, t in enumerate(targets):
        cb = t.center_base_m
        cc = t.center_camera_m
        if t.is_reachable:
            reachable_count += 1
        print(f"    Target #{idx+1} [{t.label.upper()}]:")
        print(f"      - Camera Optical Coords : Xc={cc[0]:+.3f}m, Yc={cc[1]:+.3f}m, Zc={cc[2]:+.3f}m")
        print(f"      - Robot Arm Base Coords : Xb={cb[0]:+.3f}m, Yb={cb[1]:+.3f}m, Zb={cb[2]:+.3f}m")
        print(f"      - Physical Dimensions   : Width={t.width_m*100:.1f}cm, Height={t.height_m*100:.1f}cm")
        print(f"      - Grasp Parameters      : Aperture={t.target_opening_m*100:.1f}cm, Target Force={t.target_force_n:.2f}N")
        print(f"      - Kinematic Reachable   : {Color.GREEN if t.is_reachable else Color.RED}{t.is_reachable}{Color.RESET}")

    reach_ok = reachable_count > 0
    total_checks += 1
    if print_check("Workspace Envelope Reachability Check", reach_ok, f"{reachable_count}/{len(targets)} targets within arm reach"):
        passed_checks += 1

    # --------------------------------------------------------------------------
    # Step 4: High-Resolution HUD Rendering & Snapshot Export
    # --------------------------------------------------------------------------
    print_header("4. Visual HUD Rendering & Artifact Export")
    visualizer = VisionVisualizer()
    fps = 30.0
    vis_scene = visualizer.draw_scene(
        color, targets, selected_index=0, fps=fps, system_state="PHASE_1_SCANNING"
    )
    depth_vis = visualizer.render_depth_colormap(depth, max_depth_m=1.8)

    # Combine side-by-side: RGB HUD on left (640x480), Depth Jet colormap on right (320x480)
    combined = np.hstack((vis_scene, cv2.resize(depth_vis, (vis_scene.shape[1] // 2, vis_scene.shape[0]))))
    
    cv2.imwrite(output_snapshot, combined)
    snapshot_ok = os.path.exists(output_snapshot) and os.path.getsize(output_snapshot) > 1000
    total_checks += 1
    if print_check("HUD 3D Perception Snapshot Saved", snapshot_ok, f"File: {output_snapshot} ({os.path.getsize(output_snapshot)} bytes)"):
        passed_checks += 1

    # Clean up camera
    cam.stop()

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    elapsed_s = time.time() - start_time
    print("\n" + "=" * 78)
    print(f" {Color.BOLD}PHASE 1 PERCEPTION DIAGNOSTICS SUMMARY{Color.RESET}")
    print("=" * 78)
    print(f" Total Perception Checks   : {total_checks}")
    print(f" Passed                    : {Color.GREEN}{passed_checks}{Color.RESET}")
    print(f" Failed                    : {Color.RED}{total_checks - passed_checks}{Color.RESET}")
    print(f" Execution Duration        : {elapsed_s:.3f} seconds")
    print(f" Active Vision Targets     : {len(targets)} resolved in 3D metric base space")

    if passed_checks == total_checks:
        print(f"\n {Color.BOLD}{Color.GREEN}>>> ALL PHASE 1 PERCEPTION CHECKS PASSED — 3D SCENE LOCKED! <<<{Color.RESET}\n")
        return True
    else:
        print(f"\n {Color.BOLD}{Color.RED}>>> PERCEPTION VERIFICATION ISSUES DETECTED <<<{Color.RESET}\n")
        return False


if __name__ == "__main__":
    force_mock_flag = "--mock" in sys.argv or "--real" not in sys.argv
    success = run_phase_1_perception(force_mock=force_mock_flag)
    sys.exit(0 if success else 1)
