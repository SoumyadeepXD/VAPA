
import os
import sys

try:
    from robotic_hand_pipeline import config
except ImportError:
    import config

FORCE_RANGE_BY_CLASS = {
    "mug": (1.0, 3.0),
    "bottle": (1.5, 4.0),
    "apple": (0.5, 1.5),
    "box": (2.0, 6.0),
    "can": (1.0, 2.5),
}

NOMINAL_WIDTH_M = 0.07


def estimate_force_n(label, width_m):
    base_min, base_max = FORCE_RANGE_BY_CLASS.get(label, (0.5, 2.0))

    size_ratio = max(0.5, min(2.0, width_m / NOMINAL_WIDTH_M))
    target = base_min + (base_max - base_min) * (size_ratio - 0.5) / 1.5

    target = max(config.FORCE_MIN_N, min(config.FORCE_MAX_N, target))
    return target