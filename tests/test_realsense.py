"""
VAPA Subsystem Test 1: RealSense 3D Camera & Object Spatial Localization
Tests RGB-D capture, depth-to-color alignment, and 3D pixel deprojection.
"""

import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.realsense_camera import RealSenseCamera
from vision.object_detector import ObjectDetector
from vision.spatial_3d import Spatial3DAnalyzer
from vision.visualizer_3d import VisionVisualizer


def run_vision_test(force_mock: bool = False, duration_s: float = None, headless: bool = False):
    print("=" * 70)
    print(" VAPA TEST: RealSense 3D Vision & Object Detection")
    print("=" * 70)
    print(f"Initializing camera (force_mock={force_mock}, headless={headless})...")

    cam = RealSenseCamera(force_mock=force_mock)
    detector = ObjectDetector()
    spatial = Spatial3DAnalyzer()
    visualizer = VisionVisualizer()

    print("\nCamera Intrinsics:", cam.get_intrinsics_dict())
    print("Depth scale:", cam.depth_scale)
    print("\nControls:")
    print("  'q' - Quit test")
    print("  's' - Save current snapshot")
    print("  'TAB' - Cycle selected object\n")

    selected_idx = 0
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            if duration_s is not None and (time.time() - start_time) >= duration_s:
                print(f"\nReached target duration {duration_s}s. Exiting cleanly.")
                break

            t0 = time.time()
            color, depth_m = cam.get_frames()
            if color is None or depth_m is None:
                continue

            frame_count += 1
            fps = 1.0 / max(1e-4, time.time() - t0)

            # Detect & 3D Localize
            detections = detector.detect(color, depth_m)
            targets = spatial.process_scene(detections, color, depth_m, cam)

            # Render Scene
            vis_scene = visualizer.draw_scene(
                color, targets, selected_index=selected_idx, fps=fps, system_state="VISION_TEST"
            )
            depth_vis = visualizer.render_depth_colormap(depth_m, max_depth_m=1.5)

            # Display side-by-side if not headless
            combined = np.hstack((vis_scene, cv2.resize(depth_vis, (vis_scene.shape[1] // 2, vis_scene.shape[0]))))
            if not headless:
                cv2.imshow("VAPA 3D Vision Test [Press Q to Quit]", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == 9:  # TAB key
                    if targets:
                        selected_idx = (selected_idx + 1) % len(targets)
                        print(f"Selected Target: {targets[selected_idx]}")
                elif key == ord("s"):
                    cv2.imwrite("vision_snapshot.png", combined)
                    print("Saved vision_snapshot.png")
            else:
                time.sleep(0.01)

            # Terminal log every 30 frames
            if frame_count % 30 == 0:
                print(f"[FPS: {fps:.1f}] Detected {len(targets)} 3D objects:")
                for i, t in enumerate(targets):
                    cb = t.center_base_m
                    print(f"  Target #{i+1}: {t.label.upper()} at [X:{cb[0]:.2f}, Y:{cb[1]:.2f}, Z:{cb[2]:.2f}]m (Score: {t.score:.2f})")

    finally:
        cam.stop()
        if not headless:
            cv2.destroyAllWindows()
        print("\nVision Test Complete.")


if __name__ == "__main__":
    force_mock_flag = "--mock" in sys.argv or "--real" not in sys.argv
    headless_flag = "--headless" in sys.argv
    dur = None
    if "--duration" in sys.argv:
        try:
            dur = float(sys.argv[sys.argv.index("--duration") + 1])
        except (IndexError, ValueError):
            dur = 2.0
    run_vision_test(force_mock=force_mock_flag, duration_s=dur, headless=headless_flag)
