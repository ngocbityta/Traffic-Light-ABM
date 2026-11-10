"""Checks the GeoJSON-backed Python layout and GAML signal schedule."""

import unittest
import csv
import tempfile
from pathlib import Path

from LeVanLuongIntersection.python_sim import LeVanLuongSimulator, road_lengths
from LeVanLuongIntersection.demand import generate_demand


class LeVanLuongTests(unittest.TestCase):
    def test_geometry_and_empty_network_phase(self):
        roads = road_lengths()
        self.assertEqual(len(roads), 20)
        self.assertEqual({roads[f"{arm}_in"]["kind"] for arm in ("NE", "SE", "SW", "NW")},
                         {"approach"})
        sim = LeVanLuongSimulator(flow_per_approach=0)
        self.assertEqual(sim.phase, "LVL")
        self.assertEqual(sim.step(250)["phase"], "LVL")
        self.assertEqual(sim.step(1)["phase"], "none")
        self.assertEqual(sim.step(15)["phase"], "HDT")
        self.assertEqual(sim.step(200)["phase"], "none")
        self.assertEqual(sim.step(15)["phase"], "LVL")
        self.assertEqual(sim.created, 0)
        self.assertEqual(sim.queue_time_integral, 0)

    def test_seeded_arrivals_repeat(self):
        a = LeVanLuongSimulator(seed=12)
        b = LeVanLuongSimulator(seed=12)
        self.assertEqual(a.step(500), b.step(500))

    def test_paired_csv_creates_exact_requested_vehicles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"
            digest = generate_demand(path, seed=12, horizon=200)
            self.assertEqual(digest, generate_demand(path, seed=12, horizon=200))
            with path.open(newline="") as stream:
                requests = list(csv.reader(stream))
            sim = LeVanLuongSimulator(seed=999, demand_file=path)
            sim.step(200)
            self.assertEqual(sim.created, len(requests))
            empty = Path(directory) / "empty.csv"
            empty.write_text("")
            self.assertEqual(LeVanLuongSimulator(demand_file=empty).step(200)["created"], 0)

    def test_training_clock_and_mobil_cooldown(self):
        sim = LeVanLuongSimulator(flow_per_approach=0, train_mode=True)
        sim._create_vehicle("NE", "through", False, 10, 40)
        sim.step(19)
        self.assertEqual(sim.vehicles[0].lane, 10)
        sim.step(1)
        self.assertEqual(sim.vehicles[0].lane, 10)
        sim.step(1)
        self.assertLess(sim.vehicles[0].lane, 10)
        sim.step(29)
        self.assertTrue(sim.dqn_waiting_action)
        self.assertEqual(sim.cycle, 50)
        sim.set_action(2)
        sim.step(14)
        self.assertEqual(sim.phase, "none")
        sim.step(1)
        self.assertEqual(sim.phase, "HDT")
        sim.step(35)
        self.assertTrue(sim.dqn_waiting_action)
        self.assertEqual(sim.cycle, 100)

    def test_mobil_resolves_spawn_overlap_before_cooldown(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 13, 40)
        sim._create_vehicle("NE", "through", False, 13, 40)
        sim.step(1)
        self.assertEqual({vehicle.lane for vehicle in sim.vehicles}, {0, 13})
        self.assertTrue(all(vehicle.speed > 0 for vehicle in sim.vehicles))

    def test_sequential_updates_match_gama_lane_change_at_tick_37(self):
        with tempfile.TemporaryDirectory() as directory:
            demand = Path(directory) / "seed1.csv"
            generate_demand(demand, seed=1, horizon=50)
            sim = LeVanLuongSimulator(demand_file=demand)
            sim.step(36)
            row8 = next(v for v in sim.vehicles if v.demand_row_id == 8)
            self.assertEqual(row8.lane, 1)
            sim.step()
            self.assertEqual(row8.lane, 0)

    def test_seed1_200_tick_replay_keeps_sequential_lane_map_consistent(self):
        sim = LeVanLuongSimulator(seed=1)
        result = sim.step(200)
        self.assertEqual(result["created"], 109)
        self.assertEqual(result["created"], result["passed"] + result["active"])

    def test_seed1_idm_speed_matches_gama_at_tick_68(self):
        with tempfile.TemporaryDirectory() as directory:
            demand = Path(directory) / "seed1.csv"
            generate_demand(demand, seed=1, horizon=200)
            sim = LeVanLuongSimulator(demand_file=demand)
            sim.step(68)
            row5 = next(v for v in sim.vehicles if v.demand_row_id == 5)
            self.assertAlmostEqual(row5.speed, 13.7184061842, places=5)

    def test_new_road_speed_applies_on_crossing_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            demand = Path(directory) / "seed1.csv"
            generate_demand(demand, seed=1, horizon=200)
            sim = LeVanLuongSimulator(demand_file=demand)
            sim.step(74)
            row5 = next(v for v in sim.vehicles if v.demand_row_id == 5)
            self.assertEqual(sim._road_state(row5)[0], "NE_SW")
            self.assertAlmostEqual(row5.position, 130.868232, places=4)
            self.assertAlmostEqual(row5.speed, 13.064696, delta=0.001)

    def test_next_road_leader_changes_mobil_choice_before_crossing(self):
        with tempfile.TemporaryDirectory() as directory:
            demand = Path(directory) / "seed1.csv"
            generate_demand(demand, seed=1, horizon=200)
            sim = LeVanLuongSimulator(demand_file=demand)
            sim.step(86)
            vehicles = {v.demand_row_id: v for v in sim.vehicles}
            self.assertEqual(sim._road_state(vehicles[0])[0], "NE_SE")
            self.assertEqual(vehicles[6].lane, 1)
            self.assertEqual(vehicles[10].lane, 0)
            self.assertEqual(vehicles[11].lane, 1)

if __name__ == "__main__":
    unittest.main()
