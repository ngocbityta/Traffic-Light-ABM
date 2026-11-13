import unittest

from scripts.audit_lvl_kinematics import seam_center_jump, stop_kinematics
from python_simulation.vehicle_geometry import Road


class KinematicAuditTests(unittest.TestCase):
    def test_hard_stop_flags_displacement_speed_mismatch_and_deceleration(self):
        metrics = stop_kinematics(
            displacement_m=2.551, dt=0.2, start_speed=14.222,
            end_speed=0.0, decel_limit=3.0)

        self.assertAlmostEqual(metrics["observed_speed_mps"], 12.755)
        self.assertAlmostEqual(metrics["end_speed_mismatch_mps"], 12.755)
        self.assertGreater(metrics["deceleration_mps2"], 3.0)
        self.assertTrue(metrics["exceeds_decel_limit"])

    def test_route_progress_rebase_with_continuous_world_pose_is_not_a_teleport(self):
        old_road = Road(((0, 0), (10, 0)), num_lanes=2)
        new_road = Road(((10, 0), (20, 0)), num_lanes=2)

        # Local route progress resets from 10 m on old_road to 0 m on
        # new_road; that coordinate rebase is not a world-center jump.
        jump = seam_center_jump(
            old_road, new_road, seam_progress=0.0,
            new_shift=0.4, old_shift=0.4)

        self.assertAlmostEqual(jump, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
