
import numpy as np
import pyrealsense2 as rs

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RealSenseCamera:
    def __init__(self, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT,
                 fps=config.FPS):
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(rs_config)

        # Depth-to-color alignment. Without this, a pixel in the color image
        # and the same pixel index in the depth image do NOT correspond to
        # the same physical point, since the sensors are physically offset.
        self.align = rs.align(rs.stream.color)

        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # meters per depth unit

    def get_intrinsics_dict(self):
        return {
            "fx": self.intrinsics.fx,
            "fy": self.intrinsics.fy,
            "cx": self.intrinsics.ppx,
            "cy": self.intrinsics.ppy,
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
        }

    def get_frames(self):
        """Returns (color_image BGR uint8 HxWx3, depth_image_m float32 HxW)
        or (None, None) if a frame wasn't ready.
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16, raw units
        depth_image_m = depth_raw.astype(np.float32) * self.depth_scale  # -> meters

        return color_image, depth_image_m

    def deproject_pixel(self, u, v, depth_m):
        """Convert a single pixel + depth (meters) into a 3D point (meters)
        in the camera coordinate frame, using RealSense's own intrinsics.
        """
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(u), float(v)], float(depth_m))
        return np.array(point, dtype=np.float32)  # [X, Y, Z] in meters

    def stop(self):
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