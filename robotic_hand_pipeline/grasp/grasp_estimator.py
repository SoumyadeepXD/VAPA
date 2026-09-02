
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GraspTarget:
    def __init__(self, label, center_3d_m, width_m, height_m, depth_m,
                 pixel_box, rgb_crop):
        self.label = label
        self.center_3d_m = center_3d_m
        self.width_m = width_m
        self.height_m = height_m
        self.depth_m = depth_m
        self.pixel_box = pixel_box
        self.rgb_crop = rgb_crop


def _robust_depth(depth_values):
    valid = depth_values[
        (depth_values > config.MIN_VALID_DEPTH_M) &
        (depth_values < config.MAX_VALID_DEPTH_M)
    ]
    if valid.size == 0:
        return None

    if config.DEPTH_STAT == "median":
        return float(np.median(valid))
    elif config.DEPTH_STAT == "mean":
        return float(np.mean(valid))
    elif config.DEPTH_STAT == "min":
        return float(np.min(valid))
    else:
        raise ValueError(f"Unknown DEPTH_STAT: {config.DEPTH_STAT}")


def estimate_grasp(detection, color_image, depth_image_m, camera):
    x0, y0, x1, y1 = detection.as_int_box()
    h, w = depth_image_m.shape
    x0, x1 = np.clip([x0, x1], 0, w - 1)
    y0, y1 = np.clip([y0, y1], 0, h - 1)
    if x1 <= x0 or y1 <= y0:
        return None

    depth_region = depth_image_m[y0:y1, x0:x1]
    depth_val = _robust_depth(depth_region)
    if depth_val is None:
        return None

    u_center = (x0 + x1) / 2.0
    v_center = (y0 + y1) / 2.0
    center_3d = camera.deproject_pixel(u_center, v_center, depth_val)

    left_3d = camera.deproject_pixel(x0, v_center, depth_val)
    right_3d = camera.deproject_pixel(x1, v_center, depth_val)
    top_3d = camera.deproject_pixel(u_center, y0, depth_val)
    bottom_3d = camera.deproject_pixel(u_center, y1, depth_val)

    width_m = float(np.linalg.norm(right_3d - left_3d))
    height_m = float(np.linalg.norm(bottom_3d - top_3d))

    rgb_crop = color_image[y0:y1, x0:x1].copy()

    return GraspTarget(
        label=detection.label,
        center_3d_m=center_3d,
        width_m=width_m,
        height_m=height_m,
        depth_m=depth_val,
        pixel_box=(x0, y0, x1, y1),
        rgb_crop=rgb_crop,
    )


def euclidean_distance_m(point_a, point_b):
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))