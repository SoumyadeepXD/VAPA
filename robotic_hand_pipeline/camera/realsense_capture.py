import time
import numpy as np

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    PYREALSENSE_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SyntheticRealSenseCamera:
    """Self-contained simulated RGB-D camera generating realistic scenes."""
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_scale = 0.001
        self.fx = 525.0
        self.fy = 525.0
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.start_time = time.time()

    def get_intrinsics_dict(self):
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }

    def get_frames(self):
        import cv2
        import math
        t = time.time() - self.start_time
        color = np.ones((self.height, self.width, 3), dtype=np.uint8) * 220
        cv2.rectangle(color, (0, int(self.height * 0.45)), (self.width, self.height), (180, 160, 140), -1)
        depth_m = np.ones((self.height, self.width), dtype=np.float32) * 1.2
        for r in range(int(self.height * 0.45), self.height):
            row_depth = 0.80 - (r - self.height * 0.45) / (self.height * 0.55) * 0.45
            depth_m[r, :] = row_depth

        mug_u, mug_v = int(self.width * 0.32), int(self.height * 0.62)
        mug_z = 0.45 + 0.02 * math.sin(t * 0.5)
        cv2.circle(color, (mug_u, mug_v), 34, (200, 100, 30), -1)
        cv2.circle(color, (mug_u, mug_v), 24, (240, 150, 60), -1)
        cv2.putText(color, "MUG", (mug_u - 20, mug_v - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        dist_mug = np.sqrt((x_grid - mug_u) ** 2 + (y_grid - mug_v) ** 2)
        depth_m[dist_mug <= 36] = mug_z

        bot_u, bot_v = int(self.width * 0.68), int(self.height * 0.58)
        bot_z = 0.52
        cv2.rectangle(color, (bot_u - 20, bot_v - 45), (bot_u + 20, bot_v + 45), (40, 160, 40), -1)
        cv2.putText(color, "BOTTLE", (bot_u - 25, bot_v - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        mask_bot = (x_grid >= bot_u - 22) & (x_grid <= bot_u + 22) & (y_grid >= bot_v - 65) & (y_grid <= bot_v + 45)
        depth_m[mask_bot] = bot_z

        return color, depth_m

    def deproject_pixel(self, u, v, depth_m):
        x = (float(u) - self.cx) * float(depth_m) / self.fx
        y = (float(v) - self.cy) * float(depth_m) / self.fy
        z = float(depth_m)
        return np.array([x, y, z], dtype=np.float32)

    def stop(self):
        pass


class RealSenseCamera:
    def __init__(self, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT,
                 fps=config.FPS):
        self.is_synthetic = False
        if not PYREALSENSE_AVAILABLE:
            self._init_synthetic(width, height, fps)
            return

        try:
            self.pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            self.profile = self.pipeline.start(rs_config)

            # Depth-to-color alignment
            self.align = rs.align(rs.stream.color)

            color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
        except Exception as e:
            print(f"RealSense hardware unavailable ({e}). Using synthetic camera fallback.")
            self._init_synthetic(width, height, fps)

    def _init_synthetic(self, width, height, fps):
        self.is_synthetic = True
        self.synth_cam = SyntheticRealSenseCamera(width=width, height=height, fps=fps)
        self.depth_scale = self.synth_cam.depth_scale

    def get_intrinsics_dict(self):
        if self.is_synthetic and hasattr(self, "synth_cam"):
            return self.synth_cam.get_intrinsics_dict()
        return {
            "fx": self.intrinsics.fx,
            "fy": self.intrinsics.fy,
            "cx": self.intrinsics.ppx,
            "cy": self.intrinsics.ppy,
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
        }

    def get_frames(self):
        if self.is_synthetic and hasattr(self, "synth_cam"):
            return self.synth_cam.get_frames()

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_image_m = depth_raw.astype(np.float32) * self.depth_scale

        return color_image, depth_image_m

    def deproject_pixel(self, u, v, depth_m):
        if self.is_synthetic and hasattr(self, "synth_cam"):
            return self.synth_cam.deproject_pixel(u, v, depth_m)
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(u), float(v)], float(depth_m))
        return np.array(point, dtype=np.float32)

    def stop(self):
        if self.is_synthetic:
            if hasattr(self, "synth_cam"):
                self.synth_cam.stop()
            return
        if hasattr(self, "pipeline"):
            self.pipeline.stop()


if __name__ == "__main__":
    import cv2

    cam = RealSenseCamera()
    print("Intrinsics:", cam.get_intrinsics_dict())
    print("Depth scale (m/unit):", cam.depth_scale)
    print("Press q to quit.")

    try:
        while True:
            color, depth_m = cam.get_frames()
            if color is None:
                continue

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_m, alpha=255.0 / 2.0), cv2.COLORMAP_JET
            )
            combined = np.hstack((color, depth_vis))
            cv2.imshow("Color | Depth", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()