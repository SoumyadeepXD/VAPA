import os
import sys

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from camera.realsense_capture import RealSenseCamera


class Detection:
    __slots__ = ("label", "score", "x_min", "y_min", "x_max", "y_max")

    def __init__(self, label, score, x_min, y_min, x_max, y_max):
        self.label = label
        self.score = score
        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max

    def as_int_box(self):
        return int(self.x_min), int(self.y_min), int(self.x_max), int(self.y_max)


class SimpleGeometricDetector:
    def detect(self, bgr_image, depth_image_m=None):
        detections = []
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = bgr_image.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > 50000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w > w_img * 0.85 or h > h_img * 0.85:
                continue

            roi = bgr_image[y:y+h, x:x+w]
            mean_b = float(np.mean(roi[:, :, 0]))
            mean_g = float(np.mean(roi[:, :, 1]))
            mean_r = float(np.mean(roi[:, :, 2]))

            aspect = float(h) / max(1.0, float(w))
            if mean_r > mean_g + 25 and mean_r > mean_b + 25:
                label = "apple"
            elif mean_b > mean_r + 20 and mean_b > mean_g + 20:
                label = "mug"
            elif aspect > 1.4:
                label = "bottle"
            else:
                label = "box"

            detections.append(Detection(
                label=label,
                score=0.88,
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + w),
                y_max=float(y + h),
            ))

        return detections


class ObjectDetector:
    def __init__(self, model_path=config.DETECTOR_MODEL_PATH,
                 score_threshold=config.DETECTION_SCORE_THRESHOLD,
                 max_results=config.MAX_DETECTIONS):
        self.detector = None
        self.geom_detector = None
        self.score_threshold = score_threshold
        self.max_results = max_results

        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
            try:
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = vision.ObjectDetectorOptions(
                    base_options=base_options,
                    score_threshold=score_threshold,
                    max_results=max_results,
                    running_mode=vision.RunningMode.IMAGE,
                )
                self.detector = vision.ObjectDetector.create_from_options(options)
            except Exception as e:
                print(f"Warning: Failed to load MediaPipe model ({e}). Using geometric detector.")

        if self.detector is None:
            self.geom_detector = SimpleGeometricDetector()

    def detect(self, bgr_image, depth_image_m=None):
        if self.detector is not None:
            try:
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                result = self.detector.detect(mp_image)

                detections = []
                for det in result.detections:
                    bbox = det.bounding_box
                    category = det.categories[0]
                    detections.append(Detection(
                        label=category.category_name,
                        score=category.score,
                        x_min=bbox.origin_x,
                        y_min=bbox.origin_y,
                        x_max=bbox.origin_x + bbox.width,
                        y_max=bbox.origin_y + bbox.height,
                    ))
                return detections
            except Exception as e:
                print(f"MediaPipe inference failed: {e}. Falling back to geometric detector.")

        if self.geom_detector is not None:
            raw_dets = self.geom_detector.detect(bgr_image, depth_image_m)
            return raw_dets[:self.max_results]

        return []


def draw_detections(image, detections):
    out = image.copy()
    for det in detections:
        x0, y0, x1, y1 = det.as_int_box()
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(out, f"{det.label} {det.score:.2f}", (x0, max(0, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    cam = RealSenseCamera()
    detector = ObjectDetector()

    try:
        while True:
            color, depth_m = cam.get_frames()
            if color is None:
                continue

            detections = detector.detect(color, depth_image_m=depth_m)
            vis = draw_detections(color, detections)
            cv2.imshow("Detections", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()