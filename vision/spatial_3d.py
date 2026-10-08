"""
VAPA 3D Spatial Localization & Grasp Pose Estimation
Transforms 2D RGB-D detections into metric 3D coordinates, applies coordinate frame
transformations (Camera -> Robot Base), and computes optimal grasp poses and opening widths.
"""

import math
import logging
import numpy as np

from config.system_config import (
    MIN_VALID_DEPTH_M,
    MAX_VALID_DEPTH_M,
    DEPTH_STAT_METHOD,
    CAMERA_TO_BASE_TRANSLATION_M,
    CAMERA_TO_BASE_ROTATION_DEG,
    WORKSPACE_BOUNDS_M,
    OBJECT_FORCE_MAP_N,
    FORCE_MIN_N,
    FORCE_MAX_N,
    GRIPPER_MAX_OPENING_M,
    GRIPPER_MIN_OPENING_M,
)
from vision.object_detector import Detection

logger = logging.getLogger("VAPA.Vision.Spatial3D")


class GraspTarget3D:
    """
    Complete 3D Grasp Target descriptor containing geometric, kinematic,
    and tactile force parameters for robotic arm execution.
    """
    def __init__(
        self,
        label: str,
        score: float,
        center_camera_m: np.ndarray,  # [X, Y, Z] in camera optical frame (meters)
        center_base_m: np.ndarray,    # [X, Y, Z] in robot base frame (meters)
        width_m: float,
        height_m: float,
        depth_m: float,
        approach_vector: np.ndarray,
        grasp_yaw_deg: float,
        grasp_pitch_deg: float,
        target_opening_m: float,
        target_force_n: float,
        is_reachable: bool,
        pixel_box: tuple,
        rgb_crop: np.ndarray = None,
    ):
        self.label = label
        self.score = score
        self.center_camera_m = np.asarray(center_camera_m, dtype=np.float32)
        self.center_base_m = np.asarray(center_base_m, dtype=np.float32)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.depth_m = float(depth_m)
        self.approach_vector = np.asarray(approach_vector, dtype=np.float32)
        self.grasp_yaw_deg = float(grasp_yaw_deg)
        self.grasp_pitch_deg = float(grasp_pitch_deg)
        self.target_opening_m = float(target_opening_m)
        self.target_force_n = float(target_force_n)
        self.is_reachable = bool(is_reachable)
        self.pixel_box = pixel_box
        self.rgb_crop = rgb_crop

    def __repr__(self):
        cb = self.center_base_m
        return (
            f"GraspTarget3D(label='{self.label}', base_pos=[{cb[0]:.3f}, {cb[1]:.3f}, {cb[2]:.3f}]m, "
            f"width={self.width_m*100:.1f}cm, force={self.target_force_n:.1f}N, reachable={self.is_reachable})"
        )


class Spatial3DAnalyzer:
    """
    Extracts 3D spatial geometry from RGB-D frames and transforms points into robot base coordinates.
    """
    def __init__(self, camera_translation=CAMERA_TO_BASE_TRANSLATION_M, camera_rotation_deg=CAMERA_TO_BASE_ROTATION_DEG):
        self.t_cam_base = np.asarray(camera_translation, dtype=np.float32)
        self.rot_cam_base = self._euler_to_rotation_matrix(camera_rotation_deg)
        # Build 4x4 Homogeneous Transformation Matrix T_base_cam
        self.T_base_cam = np.eye(4, dtype=np.float32)
        self.T_base_cam[:3, :3] = self.rot_cam_base
        self.T_base_cam[:3, 3] = self.t_cam_base

    def _euler_to_rotation_matrix(self, rpy_deg: np.ndarray) -> np.ndarray:
        """Converts [roll, pitch, yaw] in degrees to a 3x3 rotation matrix."""
        r, p, y = np.radians(rpy_deg)
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        # Camera Optical frame (X: Right, Y: Down, Z: Forward)
        # Robot Base frame (X: Forward, Y: Left, Z: Up)
        # Optical to Robotics Frame change: X_base = Z_cam, Y_base = -X_cam, Z_base = -Y_cam
        R_optical_to_robot = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
        R_mount = Rz @ Ry @ Rx
        return R_mount @ R_optical_to_robot

    def transform_camera_to_base(self, point_camera_m: np.ndarray) -> np.ndarray:
        """
        Transforms a 3D point in camera coordinates [X_cam, Y_cam, Z_cam]
        into robot base coordinates [X_base, Y_base, Z_base].
        """
        p_hom = np.array([point_camera_m[0], point_camera_m[1], point_camera_m[2], 1.0], dtype=np.float32)
        p_base_hom = self.T_base_cam @ p_hom
        return p_base_hom[:3]

    def _get_robust_depth(self, depth_region: np.ndarray) -> float:
        """Extracts a statistically robust depth value ignoring noise/zeros."""
        valid = depth_region[(depth_region > MIN_VALID_DEPTH_M) & (depth_region < MAX_VALID_DEPTH_M)]
        if valid.size < 5:
            return None

        if DEPTH_STAT_METHOD == "median":
            return float(np.median(valid))
        elif DEPTH_STAT_METHOD == "trimmed_mean":
            low, high = np.percentile(valid, [15, 85])
            trimmed = valid[(valid >= low) & (valid <= high)]
            return float(np.mean(trimmed)) if trimmed.size > 0 else float(np.median(valid))
        else:
            return float(np.mean(valid))

    def get_ring_median_depth(self, depth_image_m: np.ndarray, bbox: tuple, ring_margin: int = None) -> float:
        """
        Computes the robust median depth in an outer annulus / ring surrounding the bounding box.
        Essential fallback for transparent glass, shiny metals, and dark absorption holes.
        """
        x0, y0, x1, y1 = bbox
        h, w = depth_image_m.shape[:2]
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        margin = ring_margin if ring_margin is not None else max(10, int(min(bw, bh) * 0.25))

        ry0, ry1 = max(0, y0 - margin), min(h, y1 + margin)
        rx0, rx1 = max(0, x0 - margin), min(w, x1 + margin)

        if ry1 <= ry0 or rx1 <= rx0:
            return None

        ring_patch = depth_image_m[ry0:ry1, rx0:rx1].copy()
        # Zero out the interior object bounding box so only the outer perimeter is sampled
        iy0, iy1 = max(0, y0 - ry0), min(ring_patch.shape[0], y1 - ry0)
        ix0, ix1 = max(0, x0 - rx0), min(ring_patch.shape[1], x1 - rx0)
        ring_patch[iy0:iy1, ix0:ix1] = 0.0

        return self._get_robust_depth(ring_patch)

    def estimate_grasp_target(
        self, detection: Detection, color_image: np.ndarray, depth_image_m: np.ndarray, camera
    ) -> GraspTarget3D:
        """
        Computes 3D coordinates, bounding box dimensions, grasp orientation,
        and target force for a single detected object.
        """
        x0, y0, x1, y1 = detection.as_int_box()
        h, w = depth_image_m.shape[:2]
        x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
        y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))

        if x1 - x0 < 8 or y1 - y0 < 8:
            return None

        # Sample inner region (central 60% of bbox to avoid background depth bleed)
        cx_margin = int((x1 - x0) * 0.2)
        cy_margin = int((y1 - y0) * 0.2)
        inner_depth_region = depth_image_m[y0 + cy_margin : y1 - cy_margin, x0 + cx_margin : x1 - cx_margin]

        depth_val = self._get_robust_depth(inner_depth_region)
        if depth_val is None:
            # Fallback 1: full bbox
            depth_val = self._get_robust_depth(depth_image_m[y0:y1, x0:x1])
            if depth_val is None:
                # Fallback 2: Ring-median (annulus around bbox to sample supporting tabletop/background)
                depth_val = self.get_ring_median_depth(depth_image_m, (x0, y0, x1, y1))
                if depth_val is None:
                    return None

        # 3D Deprojection helper supporting camera object or intrinsics dict
        def _deproject(u_p, v_p, z_p):
            if hasattr(camera, "deproject_pixel"):
                return camera.deproject_pixel(u_p, v_p, z_p)
            elif isinstance(camera, dict):
                fx = camera.get("fx", 615.0)
                fy = camera.get("fy", 615.0)
                cx = camera.get("cx", 320.0)
                cy = camera.get("cy", 240.0)
                return np.array([(u_p - cx) * z_p / fx, (v_p - cy) * z_p / fy, z_p], dtype=np.float32)
            else:
                return np.array([(u_p - 320.0) * z_p / 615.0, (v_p - 240.0) * z_p / 615.0, z_p], dtype=np.float32)

        # 3D Deprojection of Center & Key boundary points
        u_center = (x0 + x1) / 2.0
        v_center = (y0 + y1) / 2.0
        center_camera = _deproject(u_center, v_center, depth_val)

        # Deproject left/right/top/bottom to measure physical metric size
        left_3d = _deproject(x0, v_center, depth_val)
        right_3d = _deproject(x1, v_center, depth_val)
        top_3d = _deproject(u_center, y0, depth_val)
        bottom_3d = _deproject(u_center, y1, depth_val)

        width_m = float(np.linalg.norm(right_3d - left_3d))
        height_m = float(np.linalg.norm(bottom_3d - top_3d))
        depth_m = max(0.04, min(width_m, 0.15))  # Estimated thickness

        # Transform 3D position to Robot Base Coordinate Frame
        center_base = self.transform_camera_to_base(center_camera)

        # Calculate Grasp Approach Angles
        # Base Yaw angle to align with target in the XY plane:
        grasp_yaw_deg = math.degrees(math.atan2(center_base[1], center_base[0]))
        # Grasp Pitch angle (approach from top or horizontal):
        grasp_pitch_deg = -20.0 if center_base[2] < 0.10 else 0.0

        # Approach vector normalized
        approach_vector = np.array([math.cos(math.radians(grasp_yaw_deg)), math.sin(math.radians(grasp_yaw_deg)), 0.0])

        # Target Gripper Opening Width (object width + 20mm clearance)
        target_opening_m = np.clip(width_m + 0.020, GRIPPER_MIN_OPENING_M, GRIPPER_MAX_OPENING_M)

        # Adaptive Force Calculation based on Object Class and Size
        force_range = OBJECT_FORCE_MAP_N.get(detection.label, OBJECT_FORCE_MAP_N["default"])
        # Scale force within class range according to object size
        size_norm = np.clip(width_m / 0.08, 0.0, 1.0)
        target_force_n = force_range[0] + (force_range[1] - force_range[0]) * size_norm
        target_force_n = float(np.clip(target_force_n, FORCE_MIN_N, FORCE_MAX_N))

        # Check reachability within physical workspace boundaries
        wb = WORKSPACE_BOUNDS_M
        is_reachable = (
            (wb["x_min"] <= center_base[0] <= wb["x_max"])
            and (wb["y_min"] <= center_base[1] <= wb["y_max"])
            and (wb["z_min"] <= center_base[2] <= wb["z_max"])
        )

        rgb_crop = color_image[y0:y1, x0:x1].copy() if y1 > y0 and x1 > x0 else None

        return GraspTarget3D(
            label=detection.label,
            score=detection.score,
            center_camera_m=center_camera,
            center_base_m=center_base,
            width_m=width_m,
            height_m=height_m,
            depth_m=depth_m,
            approach_vector=approach_vector,
            grasp_yaw_deg=grasp_yaw_deg,
            grasp_pitch_deg=grasp_pitch_deg,
            target_opening_m=target_opening_m,
            target_force_n=target_force_n,
            is_reachable=is_reachable,
            pixel_box=(x0, y0, x1, y1),
            rgb_crop=rgb_crop,
        )

    def process_scene(
        self, detections: list[Detection], color_image: np.ndarray, depth_image_m: np.ndarray, camera
    ) -> list[GraspTarget3D]:
        """Processes all detected objects into 3D Grasp Targets, ranked by reachability and score."""
        targets = []
        for det in detections:
            target = self.estimate_grasp_target(det, color_image, depth_image_m, camera)
            if target is not None:
                targets.append(target)

        # Sort reachable targets first, then by detection confidence score
        targets.sort(key=lambda t: (t.is_reachable, t.score), reverse=True)
        return targets

    def analyze_scene(
        self, detections: list[Detection], color_image: np.ndarray, depth_image_m: np.ndarray, camera_or_intrinsics=None
    ) -> list[GraspTarget3D]:
        """Alias for process_scene supporting camera object or intrinsics dict."""
        return self.process_scene(detections, color_image, depth_image_m, camera_or_intrinsics)


# Global alias for Phase V validation consistency
Spatial3D = Spatial3DAnalyzer

