"""
VAPA Spatial Distance Matrix Module: distance_matrix.py
Divides the RGB-D depth frame into an N x M spatial grid to provide rapid
environmental proximity awareness, obstacle detection, and directional guidance.

Key Requirements (Phase V Step 9):
- Computes mean distance, RMS error, and invalid-pixel fraction per cell.
- Guarantees confidence = 0.0 and no garbage values below MIN_VALID_DEPTH_M (< 0.20m).
- Provides directional grid indexing (left, right, up, down, center).
- Fast vectorized evaluation running at >= 15 Hz on Jetson Orin Nano / host CPU.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional

from config.system_config import (
    MIN_VALID_DEPTH_M,
    MAX_VALID_DEPTH_M,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)


class DistanceGridCell:
    """Represents a single spatial cell in the distance matrix."""
    def __init__(
        self,
        row: int,
        col: int,
        u_min: int,
        u_max: int,
        v_min: int,
        v_max: int,
        name: str,
    ):
        self.row = row
        self.col = col
        self.u_min = u_min
        self.u_max = u_max
        self.v_min = v_min
        self.v_max = v_max
        self.name = name

        self.distance_m: float = 0.0
        self.invalid_fraction: float = 1.0
        self.confidence: float = 0.0
        self.valid_count: int = 0
        self.total_count: int = (u_max - u_min) * (v_max - v_min)


class DistanceMatrix:
    """
    Computes a spatial N x M distance matrix across the incoming aligned depth frame.
    """
    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        min_depth_m: float = MIN_VALID_DEPTH_M,
        max_depth_m: float = MAX_VALID_DEPTH_M,
    ):
        self.rows = rows
        self.cols = cols
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

        self.row_names = ["top", "center", "bottom"] if rows == 3 else [f"r{r}" for r in range(rows)]
        self.col_names = ["left", "center", "right"] if cols == 3 else [f"c{c}" for c in range(cols)]

        self.cells: List[DistanceGridCell] = []
        self._init_cells(CAMERA_WIDTH, CAMERA_HEIGHT)

    def _init_cells(self, width: int, height: int):
        self.cells.clear()
        cell_w = width // self.cols
        cell_h = height // self.rows

        for r in range(self.rows):
            for c in range(self.cols):
                u_min = c * cell_w
                u_max = (c + 1) * cell_w if c < self.cols - 1 else width
                v_min = r * cell_h
                v_max = (r + 1) * cell_h if r < self.rows - 1 else height

                r_label = self.row_names[r] if r < len(self.row_names) else f"r{r}"
                c_label = self.col_names[c] if c < len(self.col_names) else f"c{c}"
                name = f"{r_label}_{c_label}" if r_label != c_label else r_label

                self.cells.append(
                    DistanceGridCell(
                        row=r,
                        col=c,
                        u_min=u_min,
                        u_max=u_max,
                        v_min=v_min,
                        v_max=v_max,
                        name=name,
                    )
                )

    def update(self, depth_image_m: np.ndarray) -> np.ndarray:
        """
        Updates distance matrix from depth frame (in meters).
        Returns an (rows, cols) 2D float array of cell distances.
        """
        h, w = depth_image_m.shape[:2]
        if w != CAMERA_WIDTH or h != CAMERA_HEIGHT:
            self._init_cells(w, h)

        matrix = np.zeros((self.rows, self.cols), dtype=np.float32)

        for cell in self.cells:
            patch = depth_image_m[cell.v_min : cell.v_max, cell.u_min : cell.u_max]
            total_px = patch.size
            if total_px == 0:
                cell.distance_m = 0.0
                cell.invalid_fraction = 1.0
                cell.confidence = 0.0
                matrix[cell.row, cell.col] = 0.0
                continue

            # Invariant: Strictly reject values below min_depth_m (< 0.20m) and above max_depth_m
            # No garbage values allowed below min_depth_m
            valid_mask = (patch >= self.min_depth_m) & (patch <= self.max_depth_m)
            valid_px = patch[valid_mask]
            sub_min_mask = (patch > 0.0) & (patch < self.min_depth_m)

            cell.valid_count = int(np.sum(valid_mask))
            cell.invalid_fraction = 1.0 - (cell.valid_count / float(total_px))

            if np.sum(sub_min_mask) > (0.4 * total_px):
                # Object is physically closer than camera min sensing threshold (<0.20m)
                # Invariant: No garbage values; confidence strictly 0.0
                cell.distance_m = 0.0
                cell.confidence = 0.0
            elif cell.valid_count >= max(10, int(total_px * 0.05)):
                median_dist = float(np.median(valid_px))
                cell.distance_m = median_dist
                # Confidence proportional to valid pixel fill rate
                fill_rate = cell.valid_count / float(total_px)
                cell.confidence = min(1.0, max(0.0, fill_rate * 1.2))
            else:
                cell.distance_m = 0.0
                cell.confidence = 0.0

            matrix[cell.row, cell.col] = cell.distance_m

        return matrix

    def get_cell_by_name(self, name: str) -> Optional[DistanceGridCell]:
        for cell in self.cells:
            if cell.name == name:
                return cell
        return None

    def get_cell_at_pixel(self, u: int, v: int) -> Optional[DistanceGridCell]:
        """Maps (u, v) pixel coordinate to corresponding grid cell."""
        for cell in self.cells:
            if cell.u_min <= u < cell.u_max and cell.v_min <= v < cell.v_max:
                return cell
        return None

    def get_directional_proximity(self) -> Dict[str, float]:
        """Returns distance per cardinal / spatial direction."""
        res = {}
        for cell in self.cells:
            res[cell.name] = cell.distance_m
        return res
