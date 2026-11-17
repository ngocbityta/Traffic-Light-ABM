"""Heavy vehicles (trucks and buses merged into one type), 16/11/2026."""
import csv
import math
import tempfile
import unittest
from pathlib import Path

from python_simulation import ssm, trajectory
from python_simulation.calibration import demand_from_counts
from python_simulation.demand import OCCUPIED, generate_demand, read_demand
from python_simulation.python_sim import (DEFAULT_HEAVY_RATIO, DEFAULT_VEHICLE_PARAMS, DT,
                                          LIGHT_VEHICLE_LENGTH, LeVanLuongSimulator,
                                          kind_from_draw, vehicle_kind)
from python_simulation.sim_config import load_sim_config
from python_simulation.vehicle_geometry import SUBLANE_WIDTH, polygons_overlap


class ParameterTests(unittest.TestCase):
    def test_values_follow_sumo_and_gama_defaults(self):
        heavy, car = DEFAULT_VEHICLE_PARAMS["heavy"], DEFAULT_VEHICLE_PARAMS["car"]
        # SUMO truck 7.1 m / bus 12 m, 2.4 / 2.5 m wide, 1.3 / 1.2 m/s^2.
        self.assertTrue(7.1 <= heavy["length"] <= 12.0)
        self.assertTrue(2.4 <= heavy["width"] <= 2.5)
        self.assertTrue(1.2 <= heavy["accel"] <= 1.3)
        # SUMO: heavy 4.0 against car 4.5 m/s^2 comfortable deceleration.
        self.assertLess(abs(heavy["decel"] - car["decel"] * 4.0 / 4.5), 0.05)
        # Same minGap as cars in SUMO; GAMA's default time headway 1.5 s.
        self.assertEqual(heavy["gap"], car["gap"])
        self.assertEqual(heavy["headway"], 1.5)
        # Same mean desired speed as cars, half the spread (SUMO speedDev).
        self.assertEqual(heavy["speed_mean"], car["speed_mean"])
        self.assertAlmostEqual(heavy["speed_sd"], car["speed_sd"] / 2)

    def test_sublanes_follow_body_width(self):
        self.assertEqual(OCCUPIED, {"motorbike": 1, "car": 2, "heavy": 3})
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", "heavy", 3, 50)
        v = sim.vehicles[0]
        self.assertEqual((v.kind, v.occupied, v.length), ("heavy", 3, 9.0))
        self.assertAlmostEqual(2 * v.half_width, DEFAULT_VEHICLE_PARAMS["heavy"]["width"])
        self.assertLessEqual(2 * v.half_width, v.occupied * SUBLANE_WIDTH)
        self.assertEqual((v.accel, v.decel, v.gap, v.headway), (1.25, 2.7, 1.0, 1.5))

    def test_type_arguments(self):
        self.assertEqual(vehicle_kind(True), "car")
        self.assertEqual(vehicle_kind(False), "motorbike")
        self.assertEqual(vehicle_kind(2), "heavy")
        with self.assertRaises(ValueError):
            vehicle_kind("bus")
        with self.assertRaises(ValueError):
            LeVanLuongSimulator(heavy_ratio=0.2, motorbike_ratio=0.9)
        with self.assertRaises(ValueError):
            LeVanLuongSimulator(vehicle_params={"heavy": {"width": 1.5}})

    def test_overrides(self):
        sim = LeVanLuongSimulator(flow_per_approach=0,
                                  vehicle_params={"heavy": {"length": 12.0, "headway": 2.0}})
        sim._create_vehicle("NE", "through", "heavy", 3, 50)
        self.assertEqual(sim.vehicles[0].length, 12.0)
        self.assertEqual(sim.vehicles[0].headway, 2.0)


class DemandTests(unittest.TestCase):
    def test_zero_share_reproduces_the_two_type_demand(self):
        """Heavy vehicles take the low end of the old car interval of the draw."""
        for draw in (0.0, 0.02, 0.1, 0.149, 0.15, 0.5, 0.99):
            old = "car" if draw < 1 - 0.85 else "motorbike"
            self.assertEqual(kind_from_draw(draw, 0.85, 0.0), old)
        self.assertEqual(kind_from_draw(0.01, 0.85, 0.03), "heavy")
        self.assertEqual(kind_from_draw(0.05, 0.85, 0.03), "car")

    def test_generated_demand_has_heavy_rows_and_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.csv"
            generate_demand(path, seed=3, horizon=3000)
            rows = read_demand(path)
            share = sum(r[2] == 2 for r in rows) / len(rows)
            self.assertTrue(0.015 < share < 0.05, share)
            sim = LeVanLuongSimulator(seed=3, demand_file=path)
            seen = set()
            for _ in range(600):
                sim.step(1)
                seen.update(v.kind for v in sim.vehicles)
            self.assertEqual(seen, {"motorbike", "car", "heavy"})
            two = Path(tmp) / "two.csv"
            generate_demand(two, seed=3, horizon=400, heavy_ratio=0.0)
            self.assertEqual({r[2] for r in read_demand(two)}, {0, 1})

    def test_invalid_type_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("1,0,3,40,2,0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_demand(path)
            with self.assertRaises(ValueError):
                LeVanLuongSimulator(demand_file=path)

    def test_sim_config_carries_the_share(self):
        self.assertEqual(load_sim_config()["heavy_ratio"], DEFAULT_HEAVY_RATIO)
        self.assertEqual(load_sim_config(config={"heavy_ratio": 0.1})["heavy_ratio"], 0.1)

    def test_counts_give_heavy_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_slot", "station_name", "motorbike", "car",
                                 "small_truck_bus", "heavy_truck_bus", "trailer"])
                writer.writerow(["7h", "A", 70, 20, 6, 4, 0])
            row = demand_from_counts(path)[0]
            self.assertAlmostEqual(row["motorbike_ratio"], 0.7)
            self.assertAlmostEqual(row["heavy_ratio"], 0.1)


class BehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A heavy-rich mix makes the checks below meet many heavy vehicles.
        cls.sim = sim = LeVanLuongSimulator(seed=5, heavy_ratio=0.10)
        cls.max_accel_excess = 0.0
        cls.heavy_speeds = []
        cls.heavy_passed = 0
        cls.contacts = []
        cls.heavy_ids = set()
        previous = {}
        for _ in range(1500):
            before = {v.creation_id: v for v in sim.vehicles}
            sim.step(1)
            now = {v.creation_id: v for v in sim.vehicles}
            cls.heavy_passed += sum(1 for vid, v in before.items()
                                    if vid not in now and v.kind == "heavy")
            for v in sim.vehicles:
                old = previous.get(v.creation_id)
                if old is not None:
                    cls.max_accel_excess = max(cls.max_accel_excess,
                                               (v.speed - old) / DT - v.accel)
                if v.kind == "heavy" and v.road == v.out_road:
                    cls.heavy_speeds.append(v.speed)
            previous = {v.creation_id: v.speed for v in sim.vehicles}
            heavies = [v for v in sim.vehicles if v.kind == "heavy"]
            cls.heavy_ids.update(v.creation_id for v in heavies)
            if not heavies:
                continue
            bodies = {v.creation_id: sim._body_polygons(v) for v in sim.vehicles}
            for h in heavies:
                for o in sim.vehicles:
                    if o is h or abs(o.position - h.position) > 30:
                        continue
                    if any(polygons_overlap(a, b) for a in bodies[h.creation_id]
                           for b in bodies[o.creation_id]):
                        cls.contacts.append((sim.cycle, h.creation_id, o.creation_id))

    def test_heavy_vehicles_cross_the_junction(self):
        self.assertGreater(self.heavy_passed, 15)
        self.assertEqual(self.sim.jam_removed, 0)
        # 10 % heavy vehicles is a stress mix: the road guard may still
        # shorten a light follower's move by a few centimetres when its
        # leader is held at the stop line in the same tick (seed 5: one
        # push of 0.10 m between two motorbikes).
        self.assertLessEqual(self.sim.road_safety_corrections, 2)
        self.assertLess(self.sim.max_road_safety_correction, 0.15)

    def test_heavy_acceleration_is_bounded(self):
        self.assertLess(self.max_accel_excess, 0.3)
        self.assertLessEqual(max(self.heavy_speeds), 60.0 / 3.6 + 1e-6)

    def test_heavy_bodies_rarely_touch(self):
        """Body contacts with heavy vehicles stay rare and short (the light
        traffic of the model has about one contact tick per 600 s)."""
        pairs = {(a, b) for _, a, b in self.contacts}
        self.assertLessEqual(len(pairs), 4, sorted(pairs))
        self.assertLessEqual(len(self.contacts), 20)

    def test_heavy_body_is_one_rigid_block_on_two_axles(self):
        v = next(v for v in self.sim.vehicles if v.kind == "heavy")
        self.assertEqual(len(self.sim._body_polygons(v)), 1)
        # AASHTO SU: front overhang 1.2 m, wheelbase 6.1 m.
        self.assertAlmostEqual(v.wheelbase, 6.1)
        self.assertAlmostEqual(v.front_overhang, 1.2)
        front, rear, _ = self.sim._axles(v)
        self.assertAlmostEqual(math.hypot(front.x - rear[0], front.y - rear[1]),
                               v.wheelbase, places=6)
        light = next(v for v in self.sim.vehicles if v.kind != "heavy")
        self.assertEqual(light.wheelbase, 0.0)
        self.assertEqual(len(self.sim._body_polygons(light)), 1)

    def test_rear_axle_off_tracks_inside_a_right_turn(self):
        """Steady turn of radius R: the rear axle runs at sqrt(R^2 - wb^2)."""
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("SW", "right", "heavy", 0, 30)
        v = sim.vehicles[0]
        worst = 0.0
        for _ in range(400):
            sim.step(1)
            if not sim.vehicles:
                break
            if v.road != v.conn_road:
                continue
            front, rear, s_front = sim._axles(v)
            path = sim._route_pose(v, sim._walk(v, s_front, -v.wheelbase))
            # Shifts grow to the left; a right turn has its inside to the right.
            side = (rear[0] - path.x) * path.ty - (rear[1] - path.y) * path.tx
            worst = max(worst, side)
        radius = 11.6
        expected = radius - math.sqrt(radius ** 2 - 6.1 ** 2)
        self.assertGreater(worst, 0.6 * expected)
        self.assertLess(worst, 2.0 * expected)
        self.assertEqual(sim.passed, 1)

    def test_too_tight_right_turn_is_rerouted(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        # SE -> NE: even the outermost sublane is tighter than the SU
        # minimum turning radius (11.6 m).
        self.assertIsNone(sim._heavy_right_lane["SE_NE"])
        sim._create_vehicle("SE", "right", "heavy", 0, 30)
        self.assertEqual(sim.vehicles[0].maneuver, "through")
        self.assertEqual(sim.heavy_turns_rerouted, 1)
        for road in ("NE_NW", "SW_SE", "NW_SW"):
            lane = sim._heavy_right_lane[road]
            self.assertIsNotNone(lane)
            shift = (lane + 1) * SUBLANE_WIDTH
            self.assertGreaterEqual(sim._path_min_radius(road, shift), 11.55)

    def test_heavy_vehicles_take_curves_slower(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "left", "car", 7, 50)
        sim._create_vehicle("NE", "left", "heavy", 7, 50)
        car, heavy = sim.vehicles
        for v in (car, heavy):
            v.position = v.approach_end + 20.0
            v.road = v.conn_road
        self.assertLess(sim._curve_speed_cap(heavy), sim._curve_speed_cap(car))

    def test_corner_reach_only_for_long_bodies(self):
        self.assertTrue(all(v.body_sweep == 0.0 for v in self.sim.vehicles
                            if v.wheelbase == 0.0))
        self.assertTrue(all(0.0 <= v.body_sweep <= 2.5 for v in self.sim.vehicles))

    def test_heavy_vehicles_do_not_evade(self):
        ad = self.sim._negotiation
        p = ad.negotiator.p
        self.assertEqual((p.evade_cut_heavy, p.evade_front_heavy), (0.0, 0.0))
        self.assertFalse({e[1] for e in ad.evade_events} & self.heavy_ids)
        self.assertGreater(ad.evade_gap_events["heavy"], 0)


class SignalTests(unittest.TestCase):
    def test_yellow_logit_uses_the_bus_class(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", "car", 3, 50)
        sim._create_vehicle("NE", "through", "heavy", 3, 50)
        car, heavy = sim.vehicles
        heavy.position = car.position
        heavy.current_lateral_shift = car.current_lateral_shift
        heavy.speed = car.speed = 10.0
        # Same place and speed: only the class term differs (Bus & LCV -1.471,
        # car -1.068) apart from the half length in the distance term.
        u_car = math.log(sim.stop_or_go_probability(car) /
                         (1 - sim.stop_or_go_probability(car)))
        u_heavy = math.log(sim.stop_or_go_probability(heavy) /
                           (1 - sim.stop_or_go_probability(heavy)))
        expected = -1.471 + 1.068 + 0.083 * (heavy.length - car.length) / 2
        self.assertAlmostEqual(u_heavy - u_car, expected, places=6)


class OutputTests(unittest.TestCase):
    def test_trajectory_and_ssm_know_the_type(self):
        sim = LeVanLuongSimulator(seed=5, heavy_ratio=0.10)
        recorder = trajectory.record_run(sim, 400, every=2)
        data = recorder.arrays()
        self.assertIn(2, set(data["kind"].tolist()))
        heavy = data["kind"] == 2
        self.assertTrue((data["is_car"][heavy] == 1).all())
        self.assertTrue(abs(data["half_width"][heavy].min() - 1.225) < 1e-9)
        names = [str(n) for n in data["movement_names"]]
        code = next(i for i, n in enumerate(names) if n.endswith(":3"))
        self.assertAlmostEqual(ssm._movement_half_width(data, code), 1.225)
        self.assertTrue(ssm._movement_is_car(data, code))
        pairs = ssm.rear_end_pairs(data)
        self.assertGreater(len(pairs["gap"]), 0)


if __name__ == "__main__":
    unittest.main()
