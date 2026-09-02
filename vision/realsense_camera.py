"""
Intel RealSense RGB-D 3D Camera Driver & Synthetic Fallback
Supports RealSense D435, D435i, D455 on Jetson Orin with depth-color alignment,
intrinsics deprojection, post-processing filters, and synthetic simulation mode.
"""

import math
import time
import logging
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False

from config.system_config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    MIN_VALID_DEPTH_M,
    MAX_VALID_DEPTH_M,
)

logger = logging.getLogger("VAPA.Vision.Camera")


class CameraIntrinsics:
    """Stores camera pinhole model intrinsics for 3D deprojection."""
    def __init__(self, width=640, height=480, fx=615.0, fy=615.0, cx=320.0, cy=240.0):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }

    def deproject_pixel_to_point(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """
        Pinhole model 3D deprojection:
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        """
        x = (float(u) - self.cx) * float(depth_m) / self.fx
        y = (float(v) - self.cy) * float(depth_m) / self.fy
        z = float(depth_m)
        return np.array([x, y, z], dtype=np.float32)


class SyntheticRealSenseCamera:
    """
    Simulates a RealSense D435 RGB-D camera when physical hardware is not connected.
    Generates synthetic 3D scenes with interactive objects (mug, bottle, box, apple).
    """
    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_scale = 0.001  # 1mm = 0.001m
        self.intrinsics = CameraIntrinsics(
            width=width, height=height, fx=525.0, fy=525.0, cx=width / 2.0, cy=height / 2.0
        )
        self.start_time = time.time()
        logger.info(f"SyntheticRealSenseCamera initialized ({width}x{height} @ {fps}fps)")

    def get_intrinsics_dict(self):
        return self.intrinsics.to_dict()

    def get_frames(self):
        """Generates synthetic RGB and 3D depth frames."""
        t = time.time() - self.start_time

        # Create tabletop background
        color = np.ones((self.height, self.width, 3), dtype=np.uint8) * 220
        # Table surface in bottom half
        cv2.rectangle(color, (0, int(self.height * 0.45)), (self.width, self.height), (180, 160, 140), -1)

        # Depth map initialized to background distance (1.2 meters)
        depth_m = np.ones((self.height, self.width), dtype=np.float32) * 1.2

        # Draw a table plane with depth gradient
        for r in range(int(self.height * 0.45), self.height):
            # Depth decreases as row increases (table extends from 0.75m to 0.35m)
            row_depth = 0.80 - (r - self.height * 0.45) / (self.height * 0.55) * 0.45
            depth_m[r, :] = row_depth

        # Simulated Objects on the table:
        # Object 1: Mug (Blue) at Center-Left
        mug_u, mug_v = int(self.width * 0.32), int(self.height * 0.62)
        mug_z = 0.45 + 0.02 * math.sin(t * 0.5)
        cv2.circle(color, (mug_u, mug_v), 34, (200, 100, 30), -1)
        cv2.circle(color, (mug_u, mug_v), 24, (240, 150, 60), -1)
        cv2.rectangle(color, (mug_u + 24, mug_v - 12), (mug_u + 42, mug_v + 12), (200, 100, 30), 4)
        cv2.putText(color, "MUG", (mug_u - 20, mug_v - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

        # Apply depth footprint for mug
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        dist_mug = np.sqrt((x_grid - mug_u) ** 2 + (y_grid - mug_v) ** 2)
        mask_mug = dist_mug <= 36
        depth_m[mask_mug] = mug_z

        # Object 2: Bottle (Green) at Center-Right
        bot_u, bot_v = int(self.width * 0.68), int(self.height * 0.58)
        bot_z = 0.52
        cv2.rectangle(color, (bot_u - 20, bot_v - 45), (bot_u + 20, bot_v + 45), (40, 160, 40), -1)
        cv2.rectangle(color, (bot_u - 10, bot_v - 65), (bot_u + 10, bot_v - 45), (30, 120, 30), -1)
        cv2.putText(color, "BOTTLE", (bot_u - 25, bot_v - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

        mask_bot = (x_grid >= bot_u - 22) & (x_grid <= bot_u + 22) & (y_grid >= bot_v - 65) & (y_grid <= bot_v + 45)
        depth_m[mask_bot] = bot_z

        # Object 3: Apple (Red) at Near-Center
        app_u, app_v = int(self.width * 0.50), int(self.height * 0.72)
        app_z = 0.38
        cv2.circle(color, (app_u, app_v), 25, (30, 30, 220), -1)
        cv2.circle(color, (app_u - 5, app_v - 5), 8, (70, 70, 255), -1)
        cv2.putText(color, "APPLE", (app_u - 22, app_v - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

        dist_app = np.sqrt((x_grid - app_u) ** 2 + (y_grid - app_v) ** 2)
        mask_app = dist_app <= 26
        depth_m[mask_app] = app_z

        # Add slight Gaussian sensor noise to depth map
        noise = np.random.normal(0, 0.002, depth_m.shape).astype(np.float32)
        depth_m = np.clip(depth_m + noise, MIN_VALID_DEPTH_M, MAX_VALID_DEPTH_M)

        return color, depth_m

    def deproject_pixel(self, u: float, v: float, depth_m: float) -> np.ndarray:
        return self.intrinsics.deproject_pixel_to_point(u, v, depth_m)

    def stop(self):
        logger.info("Synthetic camera stopped.")


class RealSenseCamera:
    """
    Hardware RealSense Camera Driver with auto-detection and graceful fallback.
    """
    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, force_mock=False):
        self.width = width
        self.height = height
        self.fps = fps
        self.is_synthetic = False
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.depth_scale = 0.001
        self.spatial_filter = None
        self.temporal_filter = None
        self.hole_filling_filter = None

        if force_mock or not PYREALSENSE_AVAILABLE:
            logger.warning("RealSense library not available or mock forced. Using SyntheticRealSenseCamera.")
            self._init_synthetic()
            return

        try:
            self._init_hardware()
        except Exception as e:
            logger.error(f"Failed to connect to physical RealSense device ({e}). Falling back to SyntheticRealSenseCamera.")
            self._init_synthetic()

    def _init_synthetic(self):
        self.is_synthetic = True
        self.synthetic_cam = SyntheticRealSenseCamera(self.width, self.height, self.fps)
        self.intrinsics = self.synthetic_cam.intrinsics
        self.depth_scale = self.synthetic_cam.depth_scale

    def _init_hardware(self):
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        rs_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(rs_config)

        # Depth-to-color alignment
        self.align = rs.align(rs.stream.color)

        # Extract intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        rs_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.rs_intrinsics = rs_intrinsics
        self.intrinsics = CameraIntrinsics(
            width=rs_intrinsics.width,
            height=rs_intrinsics.height,
            fx=rs_intrinsics.fx,
            fy=rs_intrinsics.fy,
            cx=rs_intrinsics.ppx,
            cy=rs_intrinsics.ppy,
        )

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Initialize RealSense post-processing depth filters for clean point clouds
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

        logger.info("Physical RealSense Camera connected successfully.")
        logger.info(f"Depth scale: {self.depth_scale} m/unit, Intrinsics: {self.intrinsics.to_dict()}")

    def get_intrinsics_dict(self):
        return self.intrinsics.to_dict()

    def get_frames(self):
        """
        Returns (color_bgr, depth_m)
        color_bgr: np.ndarray (H, W, 3) uint8
        depth_m: np.ndarray (H, W) float32 in meters
        """
        if self.is_synthetic:
            return self.synthetic_cam.get_frames()

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None, None

            # Apply RealSense hardware filters
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.hole_filling_filter.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_image_m = depth_raw.astype(np.float32) * self.depth_scale

            return color_image, depth_image_m
        except Exception as e:
            logger.error(f"Error reading RealSense frame: {e}")
            return None, None

    def deproject_pixel(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """
        Convert pixel coordinate (u, v) and depth (meters) to 3D point [X, Y, Z] (meters)
        in the camera coordinate frame.
        """
        if not self.is_synthetic and PYREALSENSE_AVAILABLE and hasattr(self, "rs_intrinsics"):
            point = rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, [float(u), float(v)], float(depth_m))
            return np.array(point, dtype=np.float32)
        else:
            return self.intrinsics.deproject_pixel_to_point(u, v, depth_m)

    def stop(self):
        if self.is_synthetic:
            self.synthetic_cam.stop()
        elif self.pipeline:
            try:
                self.pipeline.stop()
                logger.info("RealSense pipeline stopped.")
            except Exception as e:
                logger.warning(f"Error stopping RealSense: {e}")
