"""
VAPA Grip Profile Manager
Loads and enforces object-specific tactile grip parameters from config/grip_profiles.yaml.

Safety Invariants:
1. Grip profile is selected ONLY at the onset of grasp closure (START_GRASP).
2. Mid-grasp perception updates may ONLY LOWER the force ceiling, NEVER raise it.
3. Perception NEVER initiates motion on its own. With the camera off or no objects visible,
   the arm operates directly from EMG alone using default profile parameters.
"""

import os
import yaml
import logging

logger = logging.getLogger("VAPA.Vision.GripProfiles")

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config/grip_profiles.yaml"
)


BLOCKED_AUTO_PROFILE_CLASSES = frozenset({
    "knife", "scissors", "scissor", "blade", "dagger", "box cutter"
})


class GripProfile:
    """Represents grasping parameters for a specific object class."""
    __slots__ = (
        "label",
        "target_force_n",
        "force_ceiling_n",
        "grasp_type",
        "speed_scale",
        "aperture_ratio",
    )

    def __init__(
        self,
        label: str,
        target_force_n: float,
        force_ceiling_n: float,
        grasp_type: str = "POWER",
        speed_scale: float = 0.8,
        aperture_ratio: float = 1.0,
    ):
        self.label = str(label)
        self.target_force_n = float(target_force_n)
        self.force_ceiling_n = float(force_ceiling_n)
        self.grasp_type = str(grasp_type)
        self.speed_scale = float(speed_scale)
        self.aperture_ratio = float(aperture_ratio)

    def __repr__(self):
        return (
            f"GripProfile(label='{self.label}', target_force={self.target_force_n:.1f}N, "
            f"ceiling={self.force_ceiling_n:.1f}N, type='{self.grasp_type}', speed={self.speed_scale:.2f})"
        )


class GripProfileManager:
    """
    Manages object-specific grip profiles and enforces monotonic safety rules during grasping.
    """
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.profiles: dict[str, GripProfile] = {}
        self.is_closing = False
        self.active_profile: GripProfile = None
        self.current_force_ceiling_n: float = 7.0

        self.load_profiles(config_path)

    def load_profiles(self, path: str):
        """Loads grip profiles from YAML file."""
        if not os.path.exists(path):
            logger.warning(f"Grip profile config {path} not found. Using embedded fallback.")
            self._set_default_profiles()
            return

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

            self.profiles.clear()
            for key, val in data.items():
                if isinstance(val, dict):
                    k_lower = key.lower()
                    if k_lower in BLOCKED_AUTO_PROFILE_CLASSES:
                        logger.warning(
                            f"[SAFETY] Sharp object '{key}' in config ignored. "
                            f"Custom profiles are strictly prohibited for dangerous blades."
                        )
                        continue
                    profile = GripProfile(
                        label=key,
                        target_force_n=val.get("target_force_n", 3.5),
                        force_ceiling_n=val.get("force_ceiling_n", 7.0),
                        grasp_type=val.get("grasp_type", "POWER"),
                        speed_scale=val.get("speed_scale", 0.8),
                        aperture_ratio=val.get("aperture_ratio", 1.0),
                    )
                    self.profiles[k_lower] = profile

            if "default" not in self.profiles:
                self.profiles["default"] = GripProfile("default", 3.5, 7.0, "POWER", 0.8, 1.0)

            self.active_profile = self.profiles["default"]
            self.current_force_ceiling_n = self.active_profile.force_ceiling_n
            logger.info(f"Loaded {len(self.profiles)} grip profiles from {path}.")

        except Exception as e:
            logger.error(f"Error loading grip profiles from {path}: {e}. Using fallback defaults.")
            self._set_default_profiles()

    def _set_default_profiles(self):
        self.profiles = {
            "default": GripProfile("default", 3.5, 7.0, "POWER", 0.8, 1.0),
            "mug": GripProfile("mug", 4.5, 8.0, "CYLINDRICAL", 0.8, 0.95),
            "apple": GripProfile("apple", 3.5, 6.5, "SPHERICAL", 0.8, 0.95),
            "banana": GripProfile("banana", 2.2, 4.5, "PINCH", 0.6, 0.85),
            "egg": GripProfile("egg", 1.5, 3.0, "PINCH", 0.4, 0.80),
        }
        self.active_profile = self.profiles["default"]
        self.current_force_ceiling_n = self.active_profile.force_ceiling_n

    def get_profile(self, object_class: str = None, confidence: float = 1.0) -> GripProfile:
        """
        Retrieves matching grip profile for object_class.
        Falls back to default profile if unknown, low confidence (< 0.50), or stale.
        Dangerous objects (knife, scissors) are strictly blocked from automatic profiles.
        """
        if object_class is None or confidence < 0.50:
            return self.profiles["default"]

        key = str(object_class).strip().lower()
        if key in BLOCKED_AUTO_PROFILE_CLASSES:
            logger.warning(
                f"[SAFETY] Dangerous sharp object '{key}' detected! "
                f"Refusing automatic custom grip profile; using default profile only."
            )
            return self.profiles["default"]

        return self.profiles.get(key, self.profiles["default"])

    def on_grasp_start(self, object_class: str = None, confidence: float = 1.0) -> GripProfile:
        """
        Called ONLY when closing starts (START_GRASP). Selects the initial grip profile.
        """
        self.is_closing = True
        self.active_profile = self.get_profile(object_class, confidence)
        self.current_force_ceiling_n = self.active_profile.force_ceiling_n
        logger.info(
            f"Grasp start: selected profile '{self.active_profile.label}' "
            f"(target={self.active_profile.target_force_n:.1f}N, ceiling={self.current_force_ceiling_n:.1f}N)."
        )
        return self.active_profile

    def update_mid_grasp(self, new_object_class: str, confidence: float = 1.0) -> float:
        """
        Mid-grasp perception update handler.
        SAFETY INVARIANT: Mid-grasp updates may ONLY LOWER the force ceiling, NEVER raise it.
        Returns the active (potentially lowered) force ceiling.
        """
        if not self.is_closing or self.active_profile is None:
            return self.current_force_ceiling_n

        new_profile = self.get_profile(new_object_class, confidence)
        if new_profile.force_ceiling_n < self.current_force_ceiling_n:
            old_ceiling = self.current_force_ceiling_n
            self.current_force_ceiling_n = new_profile.force_ceiling_n
            logger.info(
                f"[SAFETY] Mid-grasp perception update lowered force ceiling from {old_ceiling:.1f}N "
                f"to {self.current_force_ceiling_n:.1f}N for reclassified object '{new_profile.label}'."
            )
        return self.current_force_ceiling_n

    def on_grasp_released(self):
        """Resets grasp state upon release/open command."""
        self.is_closing = False
        self.active_profile = self.profiles.get("default")
        if self.active_profile:
            self.current_force_ceiling_n = self.active_profile.force_ceiling_n
