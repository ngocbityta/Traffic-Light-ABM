import unittest

from python_simulation.python_sim import LeVanLuongSimulator, turn_lane_bounds


class SafeStartupIntegrationTests(unittest.TestCase):
    def test_initial_green_is_not_treated_as_red_to_green_onset(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 0, 40)
        vehicle = sim.vehicles[0]

        sim.step(1)

        self.assertGreater(vehicle.speed, 0.0)
        self.assertIsNone(sim._startup_schedulers["NE"].release_time(
            vehicle.creation_id))

    def test_green_onset_snapshot_releases_staggered_queue(self):
        sim = LeVanLuongSimulator(
            seed=1, flow_per_approach=0, green_lvl=50, green_hdt=40,
            all_red=3,
        )
        # Make the timing visible in only a few ticks while retaining the
        # production scheduler and movement path.
        for scheduler in sim._startup_schedulers.values():
            scheduler.reaction_delay = 0.0
            scheduler.discharge_headway = 0.8
        # Queue inside the through corridor; lane 0 is the right-turn zone
        # where a through vehicle must merge before it may queue.
        lane = turn_lane_bounds(sim.roads["SE_in"]["num_lanes"], 1, "through")[0]
        sim._create_vehicle("SE", "through", False, lane, 40)
        sim._create_vehicle("SE", "through", False, lane, 40)
        front, follower = sim.vehicles
        stop = sim._stop_position(front)
        front.position = stop - 12
        follower.position = stop - 20
        front.speed = follower.speed = 0.0
        # Put the arm in its actual red phase immediately before its scheduled
        # HDT onset at cycle 266.
        sim.cycle = 265
        sim.phase = "none"

        sim.step(4)
        self.assertGreater(front.position, stop - 12)
        self.assertEqual(follower.position, stop - 20)

        sim.step(1)
        self.assertGreater(follower.position, stop - 20)

    def test_gama_mode_does_not_construct_or_apply_safe_scheduler(self):
        sim = LeVanLuongSimulator(
            seed=1, flow_per_approach=0, mode="gama_2025_6_4",
        )
        self.assertFalse(hasattr(sim, "_startup_schedulers"))
        sim._create_vehicle("NE", "through", False, 0, 40)
        self.assertEqual(sim.vehicles[0].creation_id, 0)
        sim.step(1)
        self.assertEqual(sim.vehicles[0].road, "NE_in")


if __name__ == "__main__":
    unittest.main()
