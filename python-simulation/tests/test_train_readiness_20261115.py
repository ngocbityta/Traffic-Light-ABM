"""Regressions for the 15/11/2026 train-readiness fixes.

Each test pins one defect of the 14/11 audit: the stop-line seam lock, the
PYTHONHASHSEED-dependent cycle breaker, gridlock under fast phase switching,
unbounded curve speed / seam acceleration, standing-start spawning and the
approach merge deadlock.
"""
import hashlib
import math
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from python_simulation.junction_negotiation import (
    Agent, JunctionNegotiator, NegotiationParams, PathSample, Zone, ZoneMap)
from python_simulation.python_sim import DT, LeVanLuongSimulator

SIMULATION = Path(__file__).resolve().parents[1] / "simulation"


def line(p0, p1, half_width=0.35, step=0.25):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    s = np.linspace(0.0, length, int(length / step) + 1)
    pts = p0 + (p1 - p0)[None, :] * (s / length)[:, None]
    return PathSample(pts, s, half_width, length)


PATHS = {
    "N_S@0:1": line((0.0, 20.0), (0.0, -20.0)),
    "N_S@1:1": line((0.3, 20.0), (0.3, -20.0)),
    # Leaves the through path to the side after a shared first stretch.
    "N_E@0:1": line((0.0, 20.0), (12.0, 4.0)),
}


def agent(vid, movement, front, speed=0.0, turn="through", arm="N", **kw):
    base = dict(vid=vid, movement=movement, arm=arm, turn=turn, front=front,
                length=1.9, speed=speed, accel=2.0, decel=3.0, max_decel=6.0,
                min_gap=0.5, desired=11.0, motorbike=True, on_junction=front > 0,
                arm_green=True, road="N_S" if front > 0 else "N_in")
    base.update(kw)
    return Agent(**base)


def negotiator():
    return JunctionNegotiator(ZoneMap(lambda m: PATHS[m], lateral_margin=0.2))


class SeamOrderTests(unittest.TestCase):
    def test_vehicle_on_connector_leads_a_longer_one_still_on_the_approach(self):
        """A 4.5 m car on the approach whose nose is ahead of a motorbike that
        is already on the connector must follow the motorbike, not lead it."""
        bike = agent("bike", "N_S@0:1", front=0.6, road="N_S")
        car = agent("car", "N_S@1:1", front=1.4, length=4.5, motorbike=False,
                    road="N_in", on_junction=True)
        stop, _ = negotiator().decide([bike, car])
        self.assertEqual(stop["bike"][0], math.inf)
        self.assertLess(stop["car"][0], car.front)

    def test_diverging_pair_is_ordered_by_body_centre(self):
        """Side by side on diverging connectors: the short vehicle whose
        centre is ahead is not made to stop behind the long one's rear."""
        bike = agent("bike", "N_S@0:1", front=2.55, road="N_S")          # centre 1.6
        car = agent("car", "N_E@0:1", front=2.95, length=4.5, motorbike=False,
                    road="N_E", turn="left")                              # centre 0.7
        car.commit_time = 0.0
        stop, _ = negotiator().decide([bike, car])
        self.assertEqual(stop["bike"][0], math.inf)


class CycleBreakerTests(unittest.TestCase):
    @staticmethod
    def edges():
        names = "abcd"
        ag = {n: agent(n, "N_S@0:1", front=float(i), hold_time=float(i))
              for i, n in enumerate(names)}
        zone = [Zone(1.0, 2.0, 1.0, 2.0, "cross")]
        # first, second (second waits for first). a->b and b->c are physical;
        # c->d and d->a are orderings. Breaking the first cycle found must
        # not leave another one behind.
        raw = [("b", "a", "in_zone"), ("c", "b", "in_zone"),
               ("d", "c", "committed"), ("a", "d", "mutual"), ("a", "c", "mutual")]
        return [[ag[f], ag[s], list(zone), r] for f, s, r in raw]

    @staticmethod
    def has_cycle(edges):
        waits = {}
        for first, second, _, _ in edges:
            waits.setdefault(second.vid, []).append(first.vid)
        state = {}

        def visit(v):
            state[v] = 1
            for n in waits.get(v, ()):
                if state.get(n) == 1 or (state.get(n) is None and visit(n)):
                    return True
            state[v] = 2
            return False
        return any(state.get(v) is None and visit(v) for v in list(waits))

    def test_no_breakable_cycle_survives(self):
        edges = self.edges()
        self.assertTrue(self.has_cycle(edges))
        negotiator()._break_cycles(edges)
        self.assertFalse(self.has_cycle(edges))
        for first, second, _, reason in edges[:2]:
            self.assertEqual(reason, "in_zone")  # physical edges are never flipped

    def test_result_does_not_depend_on_hash_seed(self):
        """The flipped edge was chosen by iterating a set of string ids."""
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])\n"
            "from test_train_readiness_20261115 import CycleBreakerTests, negotiator\n"
            "edges = CycleBreakerTests.edges(); negotiator()._break_cycles(edges)\n"
            "print([(f.vid, s.vid, r) for f, s, _, r in edges])\n")
        outputs = set()
        for hash_seed in ("1", "2", "5", "9"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            outputs.add(subprocess.run(
                [sys.executable, "-c", script, str(SIMULATION), str(Path(__file__).parent)],
                capture_output=True, text=True, check=True, env=env).stdout)
        self.assertEqual(len(outputs), 1, outputs)


def run_hashed(seed, ticks):
    sim = LeVanLuongSimulator(seed=seed)
    digest = hashlib.sha256()
    for tick in range(ticks):
        sim.step(1)
        if tick % 50 == 49:
            for v in sim.vehicles:
                digest.update(f"{v.creation_id}:{v.position:.9f}:{v.speed:.9f}:{v.lane};".encode())
    return f"{sim.passed}:{digest.hexdigest()}"


class DeterminismTests(unittest.TestCase):
    def test_same_seed_is_identical_across_processes(self):
        """Seed 3 diverged after ~1,300 ticks between two interpreters."""
        script = ("import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])\n"
                  "from test_train_readiness_20261115 import run_hashed\n"
                  "print(run_hashed(3, 1500))\n")
        procs = [subprocess.Popen(
            [sys.executable, "-c", script, str(SIMULATION), str(Path(__file__).parent)],
            stdout=subprocess.PIPE, text=True, env=dict(os.environ, PYTHONHASHSEED=h))
            for h in ("1", "2")]
        results = {p.communicate()[0].strip() for p in procs}
        self.assertTrue(all(p.returncode == 0 for p in procs))
        self.assertEqual(len(results), 1, results)


class MinimumGreenTests(unittest.TestCase):
    def test_mask_and_illegal_switch(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0, train_mode=True)
        sim.step(50)
        self.assertTrue(sim.dqn_waiting_action)
        self.assertEqual(sim.action_mask(), (True, False, True, False))  # 10 s of LVL
        sim.set_action(2)
        sim.step(50)                      # 4 s yellow + 2 s all-red + 4 s HDT
        self.assertEqual(sim.phase, "HDT")
        self.assertEqual(sim.action_mask(), (False, False, True, False))
        sim.set_action(0)                 # illegal: becomes a hold, not an error
        sim.step(50)
        self.assertEqual(sim.phase, "HDT")
        self.assertEqual(sim.masked_actions, 1)
        self.assertEqual(sim.action_mask(), (True, False, True, False))
        sim.set_action(0)
        sim.step(50)
        self.assertEqual(sim.phase, "LVL")

    def test_min_green_zero_restores_free_switching(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0, train_mode=True,
                                  min_green=0.0)
        sim.step(50)
        sim.set_action(2)
        sim.step(50)
        self.assertEqual(sim.action_mask(), (True, False, True, False))


class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sim = LeVanLuongSimulator(seed=1)
        previous = {}
        cls.max_accel_excess = 0.0
        cls.lateral = {"right": [], "left": []}
        cls.entry_speeds = []
        worst = {}
        for _ in range(1500):
            sim.step(1)
            current = {}
            for v in sim.vehicles:
                pose = sim._geometry_roads[v.road].pose(
                    sim._road_state(v)[1], v.current_lateral_shift)
                current[v.creation_id] = (v.speed, v.road, pose.x, pose.y, pose.heading)
                old = previous.get(v.creation_id)
                if old is None:
                    if v.position >= 0.0 and v.leading_distance > 60.0:
                        cls.entry_speeds.append(v.speed)
                    continue
                cls.max_accel_excess = max(cls.max_accel_excess,
                                           (v.speed - old[0]) / DT - v.accel)
                if old[1] == v.road == v.conn_road and v.maneuver in cls.lateral:
                    turn = abs((pose.heading - old[4] + math.pi) % (2 * math.pi) - math.pi)
                    travelled = math.hypot(pose.x - old[2], pose.y - old[3])
                    key = (v.maneuver, v.creation_id)
                    worst[key] = max(worst.get(key, 0.0), travelled / DT * turn / DT)
            previous = current
        for (maneuver, _), value in worst.items():
            cls.lateral[maneuver].append(value)
        cls.sim = sim

    def test_no_acceleration_beyond_the_vehicle_limit(self):
        """Road seams applied the acceleration twice (up to 4.9 m/s^2)."""
        self.assertLess(self.max_accel_excess, 0.3)

    def test_turning_vehicles_respect_the_lateral_acceleration_cap(self):
        """Right turns ran at 10 m/s on 5-10 m radii (median 5.5 m/s^2)."""
        for maneuver, values in self.lateral.items():
            with self.subTest(maneuver=maneuver):
                self.assertGreater(len(values), 20)
                self.assertLess(float(np.percentile(values, 95)), 4.0)
                self.assertLess(max(values), 4.6)

    def test_vehicles_enter_the_model_at_speed(self):
        self.assertGreater(len(self.entry_speeds), 20)
        self.assertGreater(float(np.median(self.entry_speeds)), 8.0)

    def test_curve_cap_is_off_in_gama_parity_mode(self):
        sim = LeVanLuongSimulator(seed=1, mode="gama_2025_6_4", yellow=0.0)
        self.assertEqual(sim.curve_lat_accel, 0.0)


class LivenessTests(unittest.TestCase):
    def run_policy(self, seed, ticks, choose):
        sim = LeVanLuongSimulator(seed=seed, train_mode=True)
        current, longest = 0, 0.0
        windows, last = [], 0
        for tick in range(ticks):
            if sim.dqn_waiting_action:
                wanted = choose(sim, current)
                if sim.action_mask()[wanted]:
                    current = wanted
                sim.set_action(current)
            sim.step(1)
            longest = max(longest, max((v.junction_standstill for v in sim.vehicles),
                                       default=0.0))
            if tick % 500 == 499:
                windows.append(sim.passed - last)
                last = sim.passed
        return sim, longest, windows

    def test_switching_at_every_opportunity_does_not_gridlock(self):
        """Alternating at each decision gridlocked 4/4 seeds within 220 s."""
        sim, longest, windows = self.run_policy(1, 2500, lambda sim, cur: 2 - cur)
        self.assertEqual(sim.jam_removed, 0)
        self.assertLess(longest, 60.0)
        self.assertGreater(min(windows[1:]), 120)

    def test_random_policy_does_not_gridlock(self):
        rng = random.Random(8)
        sim, longest, windows = self.run_policy(3, 2500, lambda sim, cur: rng.choice((0, 2)))
        self.assertEqual(sim.jam_removed, 0)
        self.assertLess(longest, 60.0)
        self.assertGreater(min(windows[1:]), 120)

    def test_fixed_plan_that_gridlocked_stays_live(self):
        """Seed 2 lost the junction to the seam lock (129 -> 8 vehicles/100 s)."""
        sim = LeVanLuongSimulator(seed=2)
        last, windows = 0, []
        for tick in range(3000):
            sim.step(1)
            if tick % 500 == 499:
                windows.append(sim.passed - last)
                last = sim.passed
        self.assertEqual(sim.jam_removed, 0)
        self.assertGreater(min(windows[1:]), 180)
        self.assertEqual(sim.created, sim.passed + len(sim.vehicles))


class MergeDeadlockTests(unittest.TestCase):
    def test_standing_merger_takes_the_gap_in_front_of_a_standing_follower(self):
        """A through vehicle held at its merge buffer, with the next vehicle of
        its corridor stopped a few centimetres behind, used to wait for ever
        and to block the whole approach behind it."""
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        sim.phase = "LVL"
        sim._create_vehicle("NE", "through", False, 2, 40)
        merger = sim.vehicles[-1]
        target = sim._route_target_lane(merger)
        self.assertNotEqual(target, merger.lane)
        merger.position = sim._stop_position(merger) - 24.0
        merger.speed = 0.0
        merger.merge_wait = 30.0
        merger.time_since_lane_change = 30.0
        sim._create_vehicle("NE", "through", False, target, 40)
        follower = sim.vehicles[-1]
        follower.position = merger.position - (merger.length + follower.length) / 2 - 0.1
        follower.speed = 0.0
        for _ in range(100):
            sim.step(1)
            if merger.lane == target:
                break
        self.assertEqual(merger.lane, target)
        for _ in range(250):
            sim.step(1)
        self.assertEqual(sim.passed, 2)


if __name__ == "__main__":
    unittest.main()
