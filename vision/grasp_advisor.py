"""
VAPA Grasp Advisor & Multimodal Arbitration Engine: grasp_advisor.py
Bridges 3D spatial vision perception with robotic hand actuation and EMG intent.

Key Safety Invariants & Arbitration Rules (Phase V Step 10):
1. Profile latches at CLOSE_START (cannot oscillate or change mid-motion).
2. Mid-grasp vision updates can ONLY LOWER the force ceiling, NEVER raise it.
3. Stale perception (> 300 ms) immediately falls back to safe default profile.
4. Camera disconnected/unplugged: Arm continues operating safely from EMG alone.
"""

import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


class GraspRecommendation:
    """Represents the recommended grasp profile and physical force limits."""
    def __init__(
        self,
        profile_name: str,
        target_force_n: float,
        force_ceiling_n: float,
        pre_grasp_aperture_m: float,
        min_aperture_m: float,
        finger_synergies: Dict[str, float],
        approach_vector: list,
        is_latched: bool,
        arbitration_source: str,
    ):
        self.profile_name = profile_name
        self.target_force_n = float(target_force_n)
        self.force_ceiling_n = float(force_ceiling_n)
        self.pre_grasp_aperture_m = float(pre_grasp_aperture_m)
        self.min_aperture_m = float(min_aperture_m)
        self.finger_synergies = finger_synergies
        self.approach_vector = approach_vector
        self.is_latched = is_latched
        self.arbitration_source = arbitration_source


class GraspAdvisor:
    """
    Multimodal Grasp Advisor enforcing latching, force ceiling limits,
    stale perception timeouts, and EMG camera-unplugged fallback.
    """
    DEFAULT_PROFILE = "palmar_medium"
    STALE_PERCEPTION_TIMEOUT_SEC = 0.300  # 300 ms stale perception threshold

    def __init__(self, profiles_path: str = "config/grip_profiles.yaml"):
        self.profiles_path = REPO_ROOT / profiles_path
        self.profiles: Dict[str, Any] = {}
        self.class_to_profile: Dict[str, str] = {}
        self._load_profiles()

        # Internal state tracking
        self.state = "IDLE"  # IDLE, APPROACHING, CLOSE_START, MID_GRASP, HOLDING
        self.latched_profile: Optional[str] = None
        self.active_force_ceiling: Optional[float] = None
        self.last_update_ts: float = 0.0

    def _load_profiles(self):
        if not self.profiles_path.exists():
            # Fallback inline defaults
            self.profiles = {
                "cylindrical_power": {"default_force_min_n": 1.5, "default_force_max_n": 4.5, "force_emergency_ceiling_n": 8.0, "pre_grasp_aperture_m": 0.08, "min_aperture_m": 0.025, "finger_synergies": {"thumb": 0.7, "index": 0.85, "middle": 0.9, "ring": 0.9, "pinky": 0.9}, "approach_vector_preferred": [1, 0, 0]},
                "palmar_medium": {"default_force_min_n": 0.8, "default_force_max_n": 2.5, "force_emergency_ceiling_n": 5.5, "pre_grasp_aperture_m": 0.075, "min_aperture_m": 0.02, "finger_synergies": {"thumb": 0.65, "index": 0.75, "middle": 0.8, "ring": 0.75, "pinky": 0.7}, "approach_vector_preferred": [1, 0, 0]},
                "pinch_precision": {"default_force_min_n": 0.5, "default_force_max_n": 1.8, "force_emergency_ceiling_n": 3.5, "pre_grasp_aperture_m": 0.04, "min_aperture_m": 0.005, "finger_synergies": {"thumb": 0.85, "index": 0.85, "middle": 0.4, "ring": 0.0, "pinky": 0.0}, "approach_vector_preferred": [0.707, 0, -0.707]},
                "spherical_power": {"default_force_min_n": 0.6, "default_force_max_n": 2.0, "force_emergency_ceiling_n": 4.5, "pre_grasp_aperture_m": 0.085, "min_aperture_m": 0.03, "finger_synergies": {"thumb": 0.75, "index": 0.75, "middle": 0.8, "ring": 0.8, "pinky": 0.8}, "approach_vector_preferred": [0, 0, -1]},
                "lateral_prismatic": {"default_force_min_n": 1.2, "default_force_max_n": 5.5, "force_emergency_ceiling_n": 9.0, "pre_grasp_aperture_m": 0.09, "min_aperture_m": 0.015, "finger_synergies": {"thumb": 0.8, "index": 0.9, "middle": 0.9, "ring": 0.85, "pinky": 0.8}, "approach_vector_preferred": [0, 0, -1]},
            }
            return

        with open(self.profiles_path, "r") as f:
            data = yaml.safe_load(f)
            self.profiles = data.get("profiles", {})

        for p_name, p_data in self.profiles.items():
            for c_name in p_data.get("classes", []):
                self.class_to_profile[c_name.lower()] = p_name

    def reset(self):
        """Resets latching and internal arbitration states."""
        self.state = "IDLE"
        self.latched_profile = None
        self.active_force_ceiling = None
        self.last_update_ts = 0.0

    def advise_grasp(
        self,
        object_class: Optional[str] = None,
        confidence: float = 0.0,
        distance_m: float = 1.0,
        track_stability: float = 1.0,
        perception_timestamp: Optional[float] = None,
        camera_connected: bool = True,
        emg_activation: float = 0.0,
        current_time: Optional[float] = None,
    ) -> GraspRecommendation:
        """
        Determines the optimal grasp profile and force parameters.
        Enforces all Phase V Step 10 safety arbitration rules.
        """
        now = current_time if current_time is not None else time.time()
        p_time = perception_timestamp if perception_timestamp is not None else now

        # Rule 4: Camera Disconnected / Unplugged -> Operate from EMG alone
        if not camera_connected:
            default_p = self.profiles.get(self.DEFAULT_PROFILE, {})
            f_min = default_p.get("default_force_min_n", 0.8)
            f_max = default_p.get("default_force_max_n", 2.5)
            f_prop = f_min + (f_max - f_min) * max(0.0, min(1.0, emg_activation))
            return GraspRecommendation(
                profile_name=self.DEFAULT_PROFILE,
                target_force_n=f_prop,
                force_ceiling_n=default_p.get("force_emergency_ceiling_n", 5.5),
                pre_grasp_aperture_m=default_p.get("pre_grasp_aperture_m", 0.075),
                min_aperture_m=default_p.get("min_aperture_m", 0.020),
                finger_synergies=default_p.get("finger_synergies", {}),
                approach_vector=default_p.get("approach_vector_preferred", [1, 0, 0]),
                is_latched=False,
                arbitration_source="emg_camera_unplugged",
            )

        # Rule 3: Stale perception (> 300 ms) -> Fallback to default safe profile
        if (now - p_time) > self.STALE_PERCEPTION_TIMEOUT_SEC:
            default_p = self.profiles.get(self.DEFAULT_PROFILE, {})
            return GraspRecommendation(
                profile_name=self.DEFAULT_PROFILE,
                target_force_n=default_p.get("default_force_min_n", 0.8),
                force_ceiling_n=default_p.get("force_emergency_ceiling_n", 5.5),
                pre_grasp_aperture_m=default_p.get("pre_grasp_aperture_m", 0.075),
                min_aperture_m=default_p.get("min_aperture_m", 0.020),
                finger_synergies=default_p.get("finger_synergies", {}),
                approach_vector=default_p.get("approach_vector_preferred", [1, 0, 0]),
                is_latched=self.latched_profile is not None,
                arbitration_source="stale_perception_fallback",
            )

        # Rule 1: Profile latches at CLOSE_START
        # If distance is <= 0.08m or we are already in CLOSE_START / MID_GRASP
        is_closing = (distance_m <= 0.08) or (self.state in ("CLOSE_START", "MID_GRASP", "HOLDING"))

        if is_closing and self.latched_profile is None:
            # Latch current candidate profile at close start
            candidate = self.class_to_profile.get(str(object_class).lower(), self.DEFAULT_PROFILE)
            self.latched_profile = candidate
            self.state = "CLOSE_START"
            self.active_force_ceiling = self.profiles.get(candidate, {}).get("force_emergency_ceiling_n", 6.0)

        selected_profile_name = self.latched_profile if self.latched_profile else self.class_to_profile.get(str(object_class).lower(), self.DEFAULT_PROFILE)
        prof_data = self.profiles.get(selected_profile_name, self.profiles.get(self.DEFAULT_PROFILE, {}))

        # Base force calculation
        f_min = prof_data.get("default_force_min_n", 1.0)
        f_max = prof_data.get("default_force_max_n", 3.0)
        calc_force = f_min + (f_max - f_min) * max(0.0, min(1.0, emg_activation))
        prop_ceiling = prof_data.get("force_emergency_ceiling_n", 6.0)

        # Rule 2: Mid-grasp force ceiling rule: can ONLY LOWER, NEVER raise
        if self.state in ("CLOSE_START", "MID_GRASP", "HOLDING"):
            self.state = "MID_GRASP"
            if self.active_force_ceiling is not None:
                # Can only lower
                self.active_force_ceiling = min(self.active_force_ceiling, prop_ceiling)
            else:
                self.active_force_ceiling = prop_ceiling
        else:
            self.active_force_ceiling = prop_ceiling

        target_force = min(calc_force, self.active_force_ceiling)

        return GraspRecommendation(
            profile_name=selected_profile_name,
            target_force_n=target_force,
            force_ceiling_n=self.active_force_ceiling,
            pre_grasp_aperture_m=prof_data.get("pre_grasp_aperture_m", 0.080),
            min_aperture_m=prof_data.get("min_aperture_m", 0.020),
            finger_synergies=prof_data.get("finger_synergies", {}),
            approach_vector=prof_data.get("approach_vector_preferred", [1, 0, 0]),
            is_latched=self.latched_profile is not None,
            arbitration_source="vision_guided",
        )
