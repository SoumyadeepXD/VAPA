"""
VAPA Vision Dataset Tool: capture_dataset.py
Records real-world RGB-D captures from the arm-mounted Intel RealSense D435/D435i camera.
Saves:
1. RGB color frame (.jpg)
2. Aligned 16-bit depth frame (.png, raw millimeters)
3. Metadata JSON descriptor: distance band, lighting condition, scene ID, arm pose, intrinsics

Usage:
  python tools/capture_dataset.py --class apple --distance 0.5 --lighting bright --scene scene_01
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.realsense_camera import RealSenseCamera
from actuation.arm_controller import ArmController
from config.system_config import CAMERA_FPS, CAMERA_WIDTH, CAMERA_HEIGHT


def run_capture_session(
    object_class: str,
    distance_band_m: float,
    lighting_label: str,
    scene_id: str,
    hands_in_view: bool = False,
    background_type: str = "cluttered",
    output_dir: str = "data/raw/user_captures",
    auto_burst_count: int = 0,
    force_mock: bool = False,
):
    out_path = Path(output_dir) / object_class
    rgb_dir = out_path / "rgb"
    depth_dir = out_path / "depth"
    meta_dir = out_path / "metadata"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" VAPA REAL-TIME RGB-D DATASET CAPTURE TOOL (RealSense D435i)")
    print("=" * 70)
    print(f" Object Class     : {object_class}")
    print(f" Distance Band    : {distance_band_m:.2f} m")
    print(f" Lighting Label   : {lighting_label}")
    print(f" Scene ID         : {scene_id}")
    print(f" Hands in View    : {hands_in_view}")
    print(f" Background       : {background_type}")
    print(f" Output Location  : {out_path}")
    print(" Controls: [SPACE] Capture Frame | [B] Burst 10 Frames | [Q] Exit")
    print("=" * 70)

    camera = RealSenseCamera(force_mock=force_mock)
    intrinsics = camera.get_intrinsics_dict()

    # Optional arm controller link for joint pose logging
    try:
        arm = ArmController(force_mock=True)
    except Exception:
        arm = None

    frame_idx = len(list(rgb_dir.glob("*.jpg")))
    burst_remaining = auto_burst_count

    while True:
        color_frame, depth_frame_m = camera.get_frames()
        if color_frame is None or depth_frame_m is None:
            time.sleep(0.02)
            continue

        # Convert float depth (meters) to 16-bit uint (millimeters) for standard lossless storage
        depth_mm_u16 = np.clip(depth_frame_m * 1000.0, 0, 65535).astype(np.uint16)

        # Overlay HUD on preview
        display = color_frame.copy()
        cv2.putText(
            display,
            f"VAPA CAPTURE | {object_class.upper()} | {distance_band_m}m | {lighting_label}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display,
            f"Frames Saved: {frame_idx} | Scene: {scene_id} | [SPACE] Save",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

        do_capture = False
        if burst_remaining > 0:
            do_capture = True
            burst_remaining -= 1
            time.sleep(0.1)  # 100ms burst interval
        else:
            # Interactive window if GUI available
            if not force_mock and os.environ.get("DISPLAY"):
                cv2.imshow("VAPA D435i Capture Preview", display)
                key = cv2.waitKey(10) & 0xFF
                if key == ord(" "):
                    do_capture = True
                elif key == ord("b"):
                    burst_remaining = 10
                elif key == ord("q") or key == 27:
                    break
            else:
                # Non-GUI automatic capture mode (single test capture)
                do_capture = True

        if do_capture:
            timestamp_ms = int(time.time() * 1000)
            base_filename = f"{object_class}_{scene_id}_{distance_band_m}m_{timestamp_ms}_{frame_idx:04d}"

            rgb_path = rgb_dir / f"{base_filename}.jpg"
            depth_path = depth_dir / f"{base_filename}_depth.png"
            meta_path = meta_dir / f"{base_filename}.json"

            # 1. Save RGB JPEG
            cv2.imwrite(str(rgb_path), color_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # 2. Save 16-bit Depth PNG
            cv2.imwrite(str(depth_path), depth_mm_u16)

            # 3. Save JSON Metadata
            arm_angles = arm.get_joint_angles() if arm else {}
            metadata = {
                "object_class": object_class,
                "timestamp_ms": timestamp_ms,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "distance_band_m": distance_band_m,
                "lighting_label": lighting_label,
                "scene_id": scene_id,
                "hands_in_view": hands_in_view,
                "background_type": background_type,
                "resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
                "intrinsics": intrinsics,
                "robot_arm_joint_angles": arm_angles,
                "depth_units": "millimeters_uint16",
            }
            with open(meta_path, "w") as mf:
                json.dump(metadata, mf, indent=2)

            print(f"[Captured] Frame #{frame_idx}: {base_filename}")
            frame_idx += 1

        if force_mock or not os.environ.get("DISPLAY"):
            # Headless run finishes after sample capture
            break

    camera.stop()
    cv2.destroyAllWindows()
    print(f"\nCapture session complete. Total frames saved: {frame_idx}")
    return frame_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAPA RealSense D435i Dataset Capture")
    parser.add_argument("--class", dest="cls", type=str, default="apple", help="Object class")
    parser.add_argument("--distance", type=float, default=0.5, help="Distance band in meters (0.3, 0.5, 0.8, 1.2)")
    parser.add_argument("--lighting", type=str, default="bright", choices=["bright", "dim", "direct_sun"], help="Lighting")
    parser.add_argument("--scene", type=str, default="tabletop_01", help="Unique scene ID")
    parser.add_argument("--hands", action="store_true", help="Mark if human hands are in camera view")
    parser.add_argument("--background", type=str, default="cluttered", choices=["clean", "cluttered"])
    parser.add_argument("--burst", type=int, default=0, help="Number of frames to burst record")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock camera")
    args = parser.parse_args()

    run_capture_session(
        object_class=args.cls,
        distance_band_m=args.distance,
        lighting_label=args.lighting,
        scene_id=args.scene,
        hands_in_view=args.hands,
        background_type=args.background,
        auto_burst_count=args.burst,
        force_mock=args.mock,
    )
