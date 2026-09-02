"""
VAPA 3D Vision Visualizer & HUD Overlay
Renders RGB detections, 3D metric coordinate annotations, grasp vectors,
and depth colormaps for real-time operator feedback.
"""

import cv2
import numpy as np
from vision.spatial_3d import GraspTarget3D


class VisionVisualizer:
    """Draws real-time HUD overlays on RGB and Depth streams."""

    @staticmethod
    def draw_scene(
        color_image: np.ndarray,
        targets: list[GraspTarget3D],
        selected_index: int = 0,
        fps: float = 0.0,
        system_state: str = "IDLE",
    ) -> np.ndarray:
        out = color_image.copy()
        h, w = out.shape[:2]

        # Top Status Bar
        cv2.rectangle(out, (0, 0), (w, 38), (25, 25, 25), -1)
        state_color = (0, 255, 0) if "REACH" in system_state or "GRASP" in system_state else (0, 200, 255)
        cv2.putText(out, f"VAPA VISION HUD | STATE: {system_state}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Draw each 3D Target
        for idx, target in enumerate(targets):
            x0, y0, x1, y1 = target.pixel_box
            is_selected = idx == selected_index
            box_color = (0, 255, 0) if is_selected else ((255, 180, 0) if target.is_reachable else (80, 80, 200))
            thickness = 3 if is_selected else 2

            # Bounding Box
            cv2.rectangle(out, (x0, y0), (x1, y1), box_color, thickness)

            # Target Tag Header
            tag_text = f"#{idx+1} {target.label.upper()} ({target.score:.2f})"
            if is_selected:
                tag_text += " [LOCKED]"
            cv2.rectangle(out, (x0, max(38, y0 - 22)), (x0 + len(tag_text) * 9 + 10, max(38, y0)), box_color, -1)
            cv2.putText(
                out, tag_text, (x0 + 4, max(52, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0) if is_selected else (255, 255, 255), 1, cv2.LINE_AA
            )

            # 3D Coordinates Overlay
            cb = target.center_base_m
            coord_str = f"Base: X:{cb[0]:.2f} Y:{cb[1]:.2f} Z:{cb[2]:.2f}m"
            dim_str = f"Dim: {target.width_m*100:.1f}x{target.height_m*100:.1f}cm | F:{target.target_force_n:.1f}N"

            # Background pill for text
            cv2.rectangle(out, (x0, y1 + 2), (x0 + 210, y1 + 36), (15, 15, 15), -1)
            cv2.putText(out, coord_str, (x0 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            cv2.putText(out, dim_str, (x0 + 4, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

            # Center Crosshair & Grasp Vector
            u_c = int((x0 + x1) / 2)
            v_c = int((y0 + y1) / 2)
            cv2.drawMarker(out, (u_c, v_c), box_color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        return out

    @staticmethod
    def render_depth_colormap(depth_image_m: np.ndarray, max_depth_m: float = 1.5) -> np.ndarray:
        """Converts float32 depth map into a JET colormap for visualization."""
        clipped = np.clip(depth_image_m, 0.0, max_depth_m)
        normalized = (clipped / max_depth_m * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return colored
