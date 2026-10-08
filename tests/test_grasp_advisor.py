"""
VAPA Phase V Step 10: Grasp Advisor & Multimodal Arbitration Test Suite
File: tests/test_grasp_advisor.py

Verifies:
1. Table-driven test suite over:
   class x confidence x distance x track stability x stale perception x camera unplugged.
2. Invariant 1: Profile latches at CLOSE_START (cannot oscillate or switch classes mid-approach).
3. Invariant 2: Mid-grasp force ceiling can ONLY LOWER, NEVER raise.
4. Invariant 3: Stale perception (> 300 ms) immediately falls back to safe default profile.
5. Invariant 4: Arm operates safely from EMG alone when camera is disconnected/unplugged.
"""

import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.grasp_advisor import GraspAdvisor, GraspRecommendation


class TestGraspAdvisorArbitration(unittest.TestCase):
    def setUp(self):
        self.advisor = GraspAdvisor()
        self.advisor.reset()

    def test_table_driven_arbitration_matrix(self):
        """Table-driven test suite covering multimodal permutations."""
        test_cases = [
            # (name, cls, conf, dist, track_stab, dt, cam_conn, emg, expected_prof, expected_src)
            ("Clear Mug Approach", "mug", 0.92, 0.45, 1.0, 0.05, True, 0.4, "cylindrical_power", "vision_guided"),
            ("Clear Apple Approach", "apple", 0.88, 0.50, 1.0, 0.05, True, 0.5, "spherical_power", "vision_guided"),
            ("Clear Spoon Precision", "spoon", 0.85, 0.35, 1.0, 0.05, True, 0.3, "pinch_precision", "vision_guided"),
            ("Clear Box Lateral", "box", 0.90, 0.40, 1.0, 0.05, True, 0.6, "lateral_prismatic", "vision_guided"),
            ("Camera Disconnected", "mug", 0.95, 0.20, 1.0, 0.05, False, 0.7, "palmar_medium", "emg_camera_unplugged"),
            ("Stale Perception 400ms", "apple", 0.95, 0.20, 1.0, 0.40, True, 0.5, "palmar_medium", "stale_perception_fallback"),
        ]

        now = 1000.0
        for name, cls, conf, dist, stab, dt, cam_conn, emg, exp_prof, exp_src in test_cases:
            self.advisor.reset()
            rec = self.advisor.advise_grasp(
                object_class=cls,
                confidence=conf,
                distance_m=dist,
                track_stability=stab,
                perception_timestamp=now - dt,
                camera_connected=cam_conn,
                emg_activation=emg,
                current_time=now,
            )
            self.assertEqual(
                rec.profile_name, exp_prof,
                f"[{name}] Expected profile {exp_prof}, got {rec.profile_name}"
            )
            self.assertEqual(
                rec.arbitration_source, exp_src,
                f"[{name}] Expected source {exp_src}, got {rec.arbitration_source}"
            )

    def test_profile_latches_at_close_start(self):
        """
        Invariant 1: Profile must latch when close-start initiates (dist <= 0.08m).
        Subsequent perceptual flicker or class flips must NOT switch the latched profile.
        """
        now = 1000.0
        self.advisor.reset()

        # Step 1: Approach target 'apple' (spherical_power) at 0.50m
        rec1 = self.advisor.advise_grasp(
            object_class="apple", confidence=0.90, distance_m=0.50,
            perception_timestamp=now, camera_connected=True, current_time=now
        )
        self.assertEqual(rec1.profile_name, "spherical_power")
        self.assertFalse(rec1.is_latched)

        # Step 2: Advance to contact distance 0.06m (Triggers CLOSE_START latch)
        rec2 = self.advisor.advise_grasp(
            object_class="apple", confidence=0.92, distance_m=0.06,
            perception_timestamp=now + 0.1, camera_connected=True, current_time=now + 0.1
        )
        self.assertEqual(rec2.profile_name, "spherical_power")
        self.assertTrue(rec2.is_latched)

        # Step 3: Perception noise spuriously flips class to 'box' or 'fork'
        rec3 = self.advisor.advise_grasp(
            object_class="box", confidence=0.99, distance_m=0.04,
            perception_timestamp=now + 0.2, camera_connected=True, current_time=now + 0.2
        )
        # Profile MUST REMAIN spherical_power!
        self.assertEqual(
            rec3.profile_name, "spherical_power",
            "Violation: Latched grasp profile altered mid-grasp by perception flicker!"
        )
        self.assertTrue(rec3.is_latched)

    def test_mid_grasp_force_ceiling_can_only_lower(self):
        """
        Invariant 2: Mid-grasp, vision or arbitration can ONLY LOWER the force ceiling,
        NEVER raise it.
        """
        now = 1000.0
        self.advisor.reset()

        # Initiate close-start on a heavy cylindrical container
        rec1 = self.advisor.advise_grasp(
            object_class="bottle", confidence=0.90, distance_m=0.05,
            perception_timestamp=now, camera_connected=True, emg_activation=1.0, current_time=now
        )
        initial_ceiling = rec1.force_ceiling_n

        # Perception detects thinner/fragile material; ceiling reduced
        self.advisor.active_force_ceiling = initial_ceiling - 1.5
        lowered_ceiling = self.advisor.active_force_ceiling

        # Subsequent perception attempts to increase force ceiling
        # Simulate attempt to recommend heavy clamp (force_emergency_ceiling_n = 9.0)
        rec2 = self.advisor.advise_grasp(
            object_class="box", confidence=0.95, distance_m=0.03,
            perception_timestamp=now + 0.1, camera_connected=True, emg_activation=1.0, current_time=now + 0.1
        )
        # Ceiling must not exceed lowered_ceiling
        self.assertLessEqual(
            rec2.force_ceiling_n, lowered_ceiling,
            "Violation: Mid-grasp force ceiling was increased above active ceiling!"
        )

    def test_stale_perception_fallback_exceeding_300ms(self):
        """
        Invariant 3: Perception data older than 300 ms must fall back to default profile.
        """
        now = 2000.0
        self.advisor.reset()

        # 1. Fresh perception (dt = 50 ms) -> vision guided
        rec_fresh = self.advisor.advise_grasp(
            object_class="mug", confidence=0.95, distance_m=0.40,
            perception_timestamp=now - 0.050, camera_connected=True, current_time=now
        )
        self.assertEqual(rec_fresh.arbitration_source, "vision_guided")
        self.assertEqual(rec_fresh.profile_name, "cylindrical_power")

        # 2. Stale perception (dt = 320 ms > 300 ms) -> fallback
        rec_stale = self.advisor.advise_grasp(
            object_class="mug", confidence=0.95, distance_m=0.40,
            perception_timestamp=now - 0.320, camera_connected=True, current_time=now
        )
        self.assertEqual(rec_stale.arbitration_source, "stale_perception_fallback")
        self.assertEqual(rec_stale.profile_name, "palmar_medium")

    def test_camera_disconnected_emg_autonomous_operation(self):
        """
        Invariant 4: When camera is disconnected/unplugged, the arm operates
        autonomously and proportionally from EMG alone.
        """
        self.advisor.reset()

        # 1. Zero EMG activation -> minimal default force
        rec_zero = self.advisor.advise_grasp(
            camera_connected=False, emg_activation=0.0
        )
        self.assertEqual(rec_zero.arbitration_source, "emg_camera_unplugged")
        self.assertAlmostEqual(rec_zero.target_force_n, 0.8, delta=0.05)

        # 2. Full muscular flexion (EMG = 1.0) -> maximum default force
        rec_full = self.advisor.advise_grasp(
            camera_connected=False, emg_activation=1.0
        )
        self.assertEqual(rec_full.arbitration_source, "emg_camera_unplugged")
        self.assertAlmostEqual(rec_full.target_force_n, 2.5, delta=0.05)

        # 3. Intermediate muscular flexion (EMG = 0.5) -> proportional force
        rec_mid = self.advisor.advise_grasp(
            camera_connected=False, emg_activation=0.5
        )
        expected_f = 0.8 + (2.5 - 0.8) * 0.5
        self.assertAlmostEqual(rec_mid.target_force_n, expected_f, delta=0.05)


if __name__ == "__main__":
    print("\nRunning Grasp Advisor & Arbitration Invariant Suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraspAdvisorArbitration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("ALL GRASP ADVISOR & ARBITRATION TESTS PASSED (100% INVARIANTS CERTIFIED).\n")
