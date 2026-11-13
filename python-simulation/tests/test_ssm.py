import math
import unittest

import numpy as np

from python_simulation import ssm, trajectory
from python_simulation.python_sim import LeVanLuongSimulator


class FormulaTests(unittest.TestCase):
    def test_mttc_matches_cluster1_example(self):
        # Cum 1, example 1: D = 16 m, dv = 2 m/s, leader braking at 4 m/s^2.
        self.assertAlmostEqual(float(ssm.ttc_1d(16.0, 2.0)), 8.0)
        expected = (-2.0 + math.sqrt(4.0 + 2.0 * 4.0 * 16.0)) / 4.0
        self.assertAlmostEqual(float(ssm.mttc(16.0, 2.0, 4.0)), expected)
        self.assertAlmostEqual(expected, 2.372, places=3)

    def test_mttc_handles_no_collision_and_linear_cases(self):
        self.assertTrue(np.isinf(ssm.mttc(10.0, -1.0, 0.0)))
        self.assertTrue(np.isinf(ssm.mttc(10.0, 1.0, -1.0)))  # follower brakes enough
        self.assertAlmostEqual(float(ssm.mttc(10.0, 2.0, 0.0)), 5.0)

    def test_disc_ttc_matches_cluster1_example(self):
        # Cum 1, example 2: perpendicular approach, R = 2 + 1 m.
        t = ssm.ttc_circle([0, -15], [0, 10], 2.0, [-15, 0], [10, 0], 1.0)
        self.assertAlmostEqual(float(t), (600 - math.sqrt(7200)) / 400, places=6)

    def test_drac_and_exposure(self):
        self.assertAlmostEqual(float(ssm.drac(10.0, 4.0)), 0.8)
        tet, tit = ssm.exposure(np.array([1.3, 1.2, 1.2, 1.4, np.inf]), 0.2, 1.5)
        self.assertAlmostEqual(tet, 0.8)
        self.assertAlmostEqual(tit, 0.18)  # Cum 2, example 1


def frame(rows):
    keys = ("cycle", "vid", "road", "progress", "shift", "speed", "accel",
            "length", "half_width", "is_car")
    data = {k: np.array([r[i] for r in rows]) for i, k in enumerate(keys)}
    data["dt"] = np.array(0.2)
    return data


class RearEndTests(unittest.TestCase):
    def test_leader_is_smallest_gap_with_lateral_overlap(self):
        # follower moto at 0; a car ahead at 6 m (gap 2.8) overlapping
        # laterally; a moto at 4 m in a far sublane (no overlap).
        data = frame([
            (1, 1, 0, 0.0, 2.1, 8.0, 0.0, 1.9, 0.35, 0),
            (1, 2, 0, 6.0, 2.6, 6.0, 0.0, 4.5, 0.85, 1),
            (1, 3, 0, 4.0, 5.0, 6.0, 0.0, 1.9, 0.35, 0),
        ])
        pairs = ssm.rear_end_pairs(data)
        follower = list(pairs["vid_follower"]).index(1)
        self.assertEqual(pairs["vid_leader"][follower], 2)
        self.assertAlmostEqual(pairs["gap"][follower], 6.0 - (1.9 + 4.5) / 2)
        self.assertAlmostEqual(pairs["ttc"][follower], pairs["gap"][follower] / 2.0)


class PetTests(unittest.TestCase):
    def test_pet_from_interpolated_crossing_times(self):
        dt = 0.2
        names = np.array(["A_B@0:1", "C_D@0:1"])
        # Two motorbikes on different movements, 10 m/s, conflict at s = 20 m.
        rows = []
        for k in range(60):
            rows.append((k, 1, 0, 10.0 * k * dt, 0.0, 10.0, 0.0, 1.9, 0.35, 0, 0))
            rows.append((k, 2, 1, 10.0 * (k * dt - 2.0), 0.0, 10.0, 0.0, 1.9, 0.35, 0, 1))
        keys = ("cycle", "vid", "road", "progress", "shift", "speed", "accel",
                "length", "half_width", "is_car", "movement")
        data = {key: np.array([r[i] for r in rows]) for i, key in enumerate(keys)}
        data["movement"] = np.where(data["progress"] >= 0, data["movement"], -1)
        data.update(dt=np.array(dt), movement_names=names,
                    road_names=np.array(["A_B", "C_D"]),
                    road_lengths=np.array([50.0, 50.0]))
        events = ssm.pet_events(data, {(0, 1): [(20.0, 20.0)]})
        # Vehicle 1 rear leaves s = 20.35 + 0.95 at t = 2.13; vehicle 2
        # front reaches 20 - 0.35 - 0.95 at t = 2 + 1.87.
        self.assertEqual(len(events["pet"]), 1)
        self.assertAlmostEqual(float(events["pet"][0]), (2.0 + 1.87) - 2.13, places=6)


class IntegrationTests(unittest.TestCase):
    def test_recorded_run_has_no_body_overlap_and_summarises(self):
        sim = LeVanLuongSimulator(seed=1)
        recorder = trajectory.record_run(sim, 300)
        data = recorder.arrays()
        self.assertEqual(len(data["vid"]), len(data["x"]))
        summary = ssm.summarize(data)
        self.assertEqual(summary["body_overlap_samples"], 0)
        self.assertGreater(summary["rear_end_samples"], 0)
