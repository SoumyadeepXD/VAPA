
# --- Camera ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
DEPTH_UNITS_TO_METERS = 0.001  # RealSense depth is uint16 in millimeters by default

# --- Detection ---
DETECTOR_MODEL_PATH = "detection/exported_model/detector.tflite"
DETECTION_SCORE_THRESHOLD = 0.5
MAX_DETECTIONS = 5

OBJECT_CLASSES = [
    "mug",
    "bottle",
    "apple",
    "box",
    "can",
]

# --- Grasp estimation ---
DEPTH_STAT = "median"  # one of: median, mean, min
MIN_VALID_DEPTH_M = 0.05
MAX_VALID_DEPTH_M = 1.5

# --- Force ---
FORCE_MIN_N = 0.2
FORCE_MAX_N = 8.0

# --- Control loop ---
CONTROL_LOOP_HZ = 50
CLOSE_STEP_FORCE_N = 0.3
SLIP_CURRENT_DELTA_THRESHOLD = 0.15
MAX_GRASP_FORCE_HARD_LIMIT_N = 10.0