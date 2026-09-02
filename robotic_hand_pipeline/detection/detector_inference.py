
import os
import sys

import cv2
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


class ObjectDetector:
    def __init__(self, model_path=config.DETECTOR_MODEL_PATH,
                 score_threshold=config.DETECTION_SCORE_THRESHOLD,
                 max_results=config.MAX_DETECTIONS):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=score_threshold,
            max_results=max_results,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, bgr_image):
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

            detections = detector.detect(color)
            vis = draw_detections(color, detections)
            cv2.imshow("Detections", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()