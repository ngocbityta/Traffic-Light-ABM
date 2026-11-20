"""Speed-ups of 18/11/2026 must not change the simulation.

The full check is a per-tick hash of every vehicle field over 5 runs
(``rand1001``, ``rand1002``, ``fixed5``, ``fast7``, ``default3``; 14,000
ticks): identical before and after. These tests pin the pieces.
"""
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from python_simulation.demand import generate_demand
from python_simulation.junction_negotiation import Agent, ZoneMap
from python_simulation.python_sim import LeVanLuongSimulator, _SeamFollowIndex
from python_simulation.sim_config import load_sim_config


def heavy_sim(ticks=300, seed=11):
    cfg = dict(load_sim_config(), heavy_ratio=0.12)
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "demand.csv"
    generate_demand(path, seed, 2000, heavy_ratio=cfg["heavy_ratio"])
    sim = LeVanLuongSimulator(seed=seed, demand_file=str(path), train_mode=True, **cfg)
    sim.step(ticks)
    return sim


class SeamFollowIndexTests(unittest.TestCase):
    def test_boundary_crossing_keeps_live_candidates_in_original_order(self):
        def vehicle(road, position):
            return SimpleNamespace(arm="NE", in_road="NE_in", conn_road="NE_SW",
                                   out_road="SW_out", road=road, position=position)

        # The follower at 80 m can see a leader at exactly 120 m.
        vehicles = [vehicle("NE_in", 79.9), vehicle("NE_SW", 120.0),
                    vehicle("NE_in", 100.0), vehicle("SW_out", 105.0)]
        index = _SeamFollowIndex(vehicles)
        self.assertEqual(index.by_arm["NE"][2], [1, 2])
        vehicles[0].position = 81.0
        vehicles[0].road = "NE_SW"
        index.update(0, vehicles[0], 1)
        self.assertEqual(index.by_arm["NE"][2], [0, 1, 2])
        vehicles[1].road = "SW_out"
        index.update(1, vehicles[1], 3)
        self.assertEqual(index.by_arm["NE"][2], [0, 2])


class LaneEntryOrderTests(unittest.TestCase):
    def test_entries_are_ordered_by_creation_id_not_address(self):
        """Ties in progress were broken by ``id(vehicle)``: a memory address.
        Seed 1001 then diverged at tick 2,890 once the heap layout changed
        (any code change, or another interpreter run)."""
        sim = heavy_sim(200)
        seen = []
        original = sim._choose_lane

        def spy(vehicle, lane_maps):
            # The road indexes exist only while ``_move`` runs.
            for index in sim._road_index.values():
                seen.extend(index.entries)
            return original(vehicle, lane_maps)

        sim._choose_lane = spy
        sim.step(1)
        self.assertTrue(seen)
        for progress, key, vehicle in seen:
            self.assertEqual(key, vehicle.creation_id)


class PoseCacheTests(unittest.TestCase):
    def test_cached_pose_equals_direct_computation(self):
        sim = heavy_sim(1)
        rng = random.Random(3)
        for road in sim._geometry_roads.values():
            for _ in range(40):
                s = rng.uniform(-2.0, road.length + 2.0)
                shift = rng.uniform(0.0, 6.0)
                first = road.pose(s, shift)
                self.assertEqual(first, road._pose(s, shift))
                self.assertIs(road.pose(s, shift), first)

    def test_signed_zero_progress_is_not_shared(self):
        road = next(iter(LeVanLuongSimulator(seed=1)._geometry_roads.values()))
        self.assertEqual(math.copysign(1.0, road.pose(-0.0, 1.0).progress), -1.0)
        self.assertEqual(math.copysign(1.0, road.pose(0.0, 1.0).progress), 1.0)


class WalkManyTests(unittest.TestCase):
    def test_one_pass_equals_separate_walks(self):
        sim = heavy_sim(400)
        heavy = [v for v in sim.vehicles if v.wheelbase > 0.0]
        self.assertTrue(heavy)
        for vehicle in heavy:
            for distances in ([-1.0, -2.5, -7.0], [-7.0, -2.5], [0.0, -3.0], [1.0, 4.0]):
                self.assertEqual(
                    sim._walk_many(vehicle, vehicle.position, distances),
                    [sim._walk(vehicle, vehicle.position, d) for d in distances])

    def test_seam_pose_cache_equals_direct_computation(self):
        sim = heavy_sim(1)
        for old_road, next_road in (("NE_in", "NE_SW"), ("NE_SW", "SW_out")):
            if old_road not in sim._geometry_roads or next_road not in sim._geometry_roads:
                continue
            for shift in (0.0, 0.525, 3.1, 7.4):
                self.assertEqual(sim._continuous_seam_pose(old_road, next_road, shift),
                                 sim._continuous_seam_pose_uncached(old_road, next_road, shift))


class ZoneMapTests(unittest.TestCase):
    def test_both_orientations_are_memoised_consistently(self):
        sim = heavy_sim(1)
        zone_map = sim._negotiation.negotiator.zones
        self.assertIsInstance(zone_map, ZoneMap)
        movements = [m for m in ("NE_SW@3:1", "SE_NW@3:1", "NW_NE@1:1", "SW_SE@2:1")]
        for a in movements:
            for b in movements:
                forward = zone_map.zones(a, b)
                backward = zone_map.zones(b, a)
                self.assertEqual(backward, tuple(z.swapped() for z in forward))
                self.assertIs(zone_map.zones(a, b), forward)

    def test_corridor_cache(self):
        agent = Agent(vid="1", movement="NE_SW#1.25:0.40", arm="NE", turn="through",
                      front=0.0, length=1.9, speed=0.0, accel=2.0, decel=3.0,
                      max_decel=7.5, min_gap=1.0, desired=10.0, motorbike=True,
                      on_junction=False, arm_green=True)
        self.assertEqual(agent.corridor, "NE_SW")
        agent.movement = "SE_NW@2:1"
        self.assertEqual(agent.corridor, "SE_NW")


if __name__ == "__main__":
    unittest.main()
