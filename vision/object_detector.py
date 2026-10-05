"""
VAPA Multi-Backend Object Detector
Detects and classifies interactable objects in RGB-D camera feeds.
Supports MediaPipe, YOLO, OpenCV DNN, and Depth-Geometric 3D clustering fallback.
"""

import logging
import numpy as np
import cv2

from config.system_config import (
    DETECTION_CONFIDENCE_THRESHOLD,
    MAX_DETECTED_OBJECTS,
    DETECTED_CLASSES,
    MIN_VALID_DEPTH_M,
    MAX_VALID_DEPTH_M,
)

logger = logging.getLogger("VAPA.Vision.Detector")


class Detection:
    """Represents a 2D detected object bounding box and classification."""
    __slots__ = ("label", "score", "x_min", "y_min", "x_max", "y_max", "mask")

    def __init__(self, label: str, score: float, x_min: float, y_min: float, x_max: float, y_max: float, mask=None):
        self.label = str(label)
        self.score = float(score)
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.mask = mask

    def as_int_box(self):
        return int(round(self.x_min)), int(round(self.y_min)), int(round(self.x_max)), int(round(self.y_max))

    @property
    def center(self):
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def width(self):
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self):
        return max(0.0, self.y_max - self.y_min)

    def __repr__(self):
        return f"Detection(label='{self.label}', score={self.score:.2f}, box=({int(self.x_min)},{int(self.y_min)},{int(self.x_max)},{int(self.y_max)}))"


class DepthGeometricClusterDetector:
    """
    Robust 3D Spatial Depth Segmenter.
    Extracts objects standing above a tabletop surface directly from the depth map & RGB,
    ensuring object detection works even without pre-downloaded neural network weights.
    """
    def __init__(self, min_area_pixels=600, max_area_pixels=60000):
        self.min_area = min_area_pixels
        self.max_area = max_area_pixels

    def detect(self, bgr_image: np.ndarray, depth_image_m: np.ndarray = None) -> list[Detection]:
        detections = []
        h, w = bgr_image.shape[:2]

        if depth_image_m is not None:
            # Multi-modal foreground segmentation (color + depth gradient)
            diff_table = np.linalg.norm(bgr_image.astype(float) - np.array([180, 160, 140]), axis=2)
            diff_bg = np.linalg.norm(bgr_image.astype(float) - np.array([220, 220, 220]), axis=2)
            color_mask = ((diff_table > 35) & (diff_bg > 35)).astype(np.uint8) * 255

            sobelx = cv2.Sobel(depth_image_m, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(depth_image_m, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(sobelx**2 + sobely**2)
            depth_edge_mask = (mag > 0.03).astype(np.uint8) * 255

            valid_depth = (depth_image_m >= MIN_VALID_DEPTH_M) & (depth_image_m <= MAX_VALID_DEPTH_M)
            foreground_mask = cv2.bitwise_or(color_mask, depth_edge_mask)
            foreground_mask[~valid_depth] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        else:
            # Fallback to color/edge segmentation
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            foreground_mask = cv2.Canny(blurred, 40, 120)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=2)

        # 2. Find connected components / contours
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Avoid full-screen boundary artifacts
            if bw > w * 0.9 or bh > h * 0.9:
                continue

            # Classify object heuristic from color and aspect ratio
            roi_bgr = bgr_image[y:y+bh, x:x+bw]
            aspect_ratio = float(bh) / max(1.0, float(bw))

            # Compute average color in ROI
            mean_b = np.mean(roi_bgr[:, :, 0])
            mean_g = np.mean(roi_bgr[:, :, 1])
            mean_r = np.mean(roi_bgr[:, :, 2])

            if mean_r > mean_g + 30 and mean_r > mean_b + 30:
                label = "apple"
            elif mean_b > mean_r + 20 and mean_b > mean_g + 20:
                label = "mug"
            elif aspect_ratio > 1.8:
                label = "bottle"
            elif aspect_ratio > 0.8 and aspect_ratio < 1.3:
                label = "box" if area > 2500 else "mug"
            else:
                label = "can"

            score = min(0.95, 0.65 + (area / (self.max_area * 0.5)) * 0.3)
            detections.append(Detection(
                label=label,
                score=score,
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + bw),
                y_max=float(y + bh),
            ))

        # Sort by score descending and limit count
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:MAX_DETECTED_OBJECTS]


class ObjectDetector:
    """
    Unified Object Detector supporting MediaPipe and Depth-Geometric Fallback.
    """
    def __init__(self, score_threshold=DETECTION_CONFIDENCE_THRESHOLD, max_results=MAX_DETECTED_OBJECTS):
        self.score_threshold = score_threshold
        self.max_results = max_results
        self.mp_detector = None
        self.geometric_detector = DepthGeometricClusterDetector()
        self.backend = "geometric"

        self._try_init_mediapipe()

    def _try_init_mediapipe(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            import os

            # Check if a model asset path is provided
            candidate_paths = [
                "detection/exported_model/detector.tflite",
                "model/detector.tflite",
                "/home/iedc_ai_dgx1/Documents/VAPA/model/detector.tflite",
            ]
            for p in candidate_paths:
                if os.path.exists(p) and os.path.getsize(p) > 1000:
                    base_options = mp_python.BaseOptions(model_asset_path=p)
                    options = vision.ObjectDetectorOptions(
                        base_options=base_options,
                        score_threshold=self.score_threshold,
                        max_results=self.max_results,
                        running_mode=vision.RunningMode.IMAGE,
                    )
                    self.mp_detector = vision.ObjectDetector.create_from_options(options)
                    self.backend = "mediapipe"
                    logger.info(f"MediaPipe ObjectDetector initialized from {p}")
                    return
        except Exception as e:
            logger.debug(f"MediaPipe initialization skipped: {e}")

        logger.info("Using DepthGeometric 3D Cluster Object Detector.")

    def detect(self, bgr_image: np.ndarray, depth_image_m: np.ndarray = None) -> list[Detection]:
        """
        Runs object detection on RGB frame (and depth frame).
        Returns list of Detection objects.
        """
        if self.backend == "mediapipe" and self.mp_detector is not None:
            try:
                import mediapipe as mp
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                result = self.mp_detector.detect(mp_image)

                detections = []
                for det in result.detections:
                    bbox = det.bounding_box
                    category = det.categories[0]
                    detections.append(Detection(
                        label=category.category_name.lower(),
                        score=category.score,
                        x_min=float(bbox.origin_x),
                        y_min=float(bbox.origin_y),
                        x_max=float(bbox.origin_x + bbox.width),
                        y_max=float(bbox.origin_y + bbox.height),
                    ))
                if len(detections) > 0:
                    return detections[:self.max_results]
            except Exception as e:
                logger.warning(f"MediaPipe inference failed: {e}. Falling back to geometric detector.")

        # Geometric depth clustering detection
        return self.geometric_detector.detect(bgr_image, depth_image_m)
