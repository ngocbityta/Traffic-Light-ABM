"""Vehicles must not stand at the stop line on green without a reason.

Regressions for the 16/11/2026 rework of the yielding rules. Each test pins
one way a vehicle used to be held although advancing was safe:

* a "leader" across the stop-line seam that is in another lateral band;
* a vehicle abreast on a diverging connector;
* a leftover of the previous phase that leaves the shared conflict zone
  long before the waiting vehicle can reach it;
* the whole stream closed because one sublane of it waits in the box.
"""
import math
import unittest

import numpy as np

from python_simulation.junction_negotiation import (
    Agent, JunctionNegotiator, PathSample, ZoneMap)
from python_simulation.python_sim import LeVanLuongSimulator


def line(p0, p1, half_width=0.35, step=0.25):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    s = np.linspace(0.0, length, int(length / step) + 1)
    pts = p0 + (p1 - p0)[None, :] * (s / length)[:, None]
    return PathSample(pts, s, half_width, length)


PATHS = {
    "N_S@0:1": line((0.0, 20.0), (0.0, -20.0)),
    "N_S@1:1": line((0.3, 20.0), (0.3, -20.0)),
    "N_E@0:1": line((0.0, 20.0), (12.0, 4.0)),
    "E_W@0:1": line((20.0, 0.0), (-20.0, 0.0)),
}


def agent(vid, movement, front, speed=0.0, turn="through", arm="N", **kw):
    base = dict(vid=vid, movement=movement, arm=arm, turn=turn, front=front,
                length=1.9, speed=speed, accel=2.0, decel=3.0, max_decel=6.0,
                min_gap=0.5, desired=11.0, motorbike=True, on_junction=front > 0,
                arm_green=True, road=movement.split("@")[0] if front > 0 else f"{arm}_in")
    base.update(kw)
    return Agent(**base)


def negotiator():
    return JunctionNegotiator(ZoneMap(lambda m: PATHS[m], lateral_margin=0.2))


class SeamLeaderTests(unittest.TestCase):
    def pair(self, bike_shift):
        car = agent("car", "N_S@1:1", front=-0.2, length=4.5, motorbike=False,
                    half_width=0.85, connector_shift=0.0, connector_span=(0.0, 0.0))
        bike = agent("bike", "N_S@0:1", front=2.0, half_width=0.35,
                     connector_shift=bike_shift,
                     connector_span=(bike_shift, bike_shift))
        return car, bike

    def test_vehicle_in_another_lateral_band_is_not_a_leader(self):
        """A car at the line was held behind a motorbike standing one
        sublane to its side: its conflict band is widened for the sweep of
        its corners, the bodies are 0.4 m apart."""
        car, bike = self.pair(1.6)           # needs 0.85 + 0.35 + 0.15
        stop, _ = negotiator().decide([car, bike])
        self.assertEqual(stop["car"][0], math.inf)

    def test_vehicle_in_the_same_band_still_leads(self):
        car, bike = self.pair(1.0)
        stop, _ = negotiator().decide([car, bike])
        self.assertLess(stop["car"][0], bike.back)

    def test_band_being_entered_by_a_lateral_move_counts(self):
        car, bike = self.pair(1.6)
        bike.connector_span = (0.9, 1.6)     # still moving in from the car's side
        stop, _ = negotiator().decide([car, bike])
        self.assertLess(stop["car"][0], bike.back)


class DivergeTests(unittest.TestCase):
    def test_vehicle_abreast_on_a_diverging_connector_does_not_hold_the_line(self):
        """A through car at the line beside a left turner that has just
        entered: paths separate, bodies are apart sideways."""
        through = agent("through", "N_S@0:1", front=-0.3, length=4.5, motorbike=False,
                        half_width=0.85, xy=(0.0, 22.5), heading=(0.0, -1.0))
        left = agent("left", "N_E@0:1", front=1.0, turn="left", half_width=0.35,
                     xy=(2.4, 19.9), heading=(0.6, -0.8))
        left.commit_time = 0.0
        stop, _ = negotiator().decide([through, left])
        self.assertEqual(stop["through"][0], math.inf)

    def test_vehicle_in_front_on_the_shared_stretch_is_followed(self):
        through = agent("through", "N_S@0:1", front=-0.3, length=4.5, motorbike=False,
                        half_width=0.85, xy=(0.0, 22.5), heading=(0.0, -1.0))
        left = agent("left", "N_E@0:1", front=1.0, turn="left", half_width=0.35,
                     xy=(0.4, 19.9), heading=(0.6, -0.8))
        left.commit_time = 0.0
        stop, _ = negotiator().decide([through, left])
        self.assertLess(stop["through"][0], math.inf)


class LeftoverTests(unittest.TestCase):
    def scene(self, leftover_speed):
        green = agent("g", "N_S@0:1", front=-1.0)                 # zone ~20 m ahead
        leftover = agent("c", "E_W@0:1", front=20.4, speed=leftover_speed, arm="E",
                         arm_green=False, on_junction=True, road="E_W")
        neg = negotiator()
        stop, _ = neg.decide([green, leftover])
        return neg, stop

    def test_leftover_about_to_leave_the_zone_holds_nobody(self):
        """A leftover inside the shared zone used to mean "wait" outright,
        however far the green vehicle still was from that zone."""
        neg, stop = self.scene(8.0)
        self.assertEqual(stop["g"][0], math.inf)
        self.assertNotIn("g", neg.clearing_holds)

    def test_leftover_standing_in_the_zone_is_waited_for(self):
        neg, stop = self.scene(0.0)
        self.assertLess(stop["g"][0], 20.0)
        self.assertIn("g", neg.clearing_holds)


def green_standing(seed, ticks):
    """Vehicle-seconds a front-row vehicle stands on green without a grant,
    split by whether the keep-clear rule or a stop target holds it."""
    sim = LeVanLuongSimulator(seed=seed)
    green_ticks = {}
    held = {"box": 0, "seam": 0, "total": 0}
    for _ in range(ticks):
        sim.step(1)
        adapter = sim._negotiation
        reasons = {}
        for d in adapter.negotiator.last_decisions:
            reasons.setdefault(d.second, set()).add(d.reason)
        for arm in ("NE", "SE", "SW", "NW"):
            green_ticks[arm] = green_ticks.get(arm, 0) + 1 if sim._arm_is_green(arm) else 0
        for v in sim.vehicles:
            if (v.road != v.in_road or v.speed > 0.3 or not sim._green_for(v) or
                    v.leading_distance <= 3.0 or green_ticks[v.arm] <= 15 or
                    v.lane != sim._route_target_lane(v) or
                    v.approach_end - v.position > 30.0 or
                    v.creation_id in sim._junction_grants):
                continue
            held["total"] += 1
            if str(v.creation_id) in adapter._last_blocked_by_clearing:
                held["box"] += 1
            elif reasons.get(str(v.creation_id), set()) <= {"seam_follow", "diverge"}:
                held["seam"] += 1
    return sim, {k: n * 0.2 for k, n in held.items()}


class BlockedMajorTests(unittest.TestCase):
    def test_major_blocked_by_a_third_vehicle_does_not_hold_a_left_turner(self):
        """Ring of seed 2 (random policy): a through vehicle that stands
        because a third vehicle blocks it used to "reclaim" its turn against
        a left turner, and the left turner was what blocked the third."""
        neg = negotiator()
        major = agent("major", "N_S@0:1", front=18.6, hold_time=5.0)
        left = agent("left", "E_W@0:1", front=17.6, arm="E", turn="left",
                     road="E_W", on_junction=True)
        neg._hard_held = {"major": [agent("third", "N_S@1:1", front=8.0)]}
        z = neg.zones.zones(left.movement, major.movement)[0]
        self.assertEqual(neg.zone_verdict(left, major, z, None, {}), ("ok", "forced"))
        neg._hard_held = {}
        self.assertEqual(neg.zone_verdict(left, major, z, None, {})[0], "block")


class CornerSweepTests(unittest.TestCase):
    """The conflict band of a car is widened by the corner sweep of its body
    on the path's own curvature, not by a constant 0.75 m (16/11/2026)."""

    @classmethod
    def setUpClass(cls):
        cls.sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        cls.adapter = cls.sim._negotiation

    def test_straight_path_has_almost_no_sweep(self):
        for road in ("NE_SW", "SW_NE"):
            self.assertLess(self.adapter._sweep(road, 3.0, 4.5), 0.2)

    def test_tight_right_turn_sweeps_more_at_the_curb(self):
        inner = self.adapter._sweep("SE_NE", 0.5, 4.5)
        outer = self.adapter._sweep("SE_NE", 3.7, 4.5)
        self.assertGreater(inner, outer)
        self.assertGreater(inner, 0.45)
        self.assertLessEqual(inner, self.adapter.sweep_car)

    def test_motorbike_is_not_swept(self):
        self.sim._create_vehicle("NE", "through", False, 3, 40)
        bike = self.sim.vehicles[-1]
        band = self.adapter._band_path(bike, "NE_SW", False)
        self.sim._create_vehicle("NE", "through", True, 3, 40)
        car = self.sim.vehicles[-1]
        car_band = self.adapter._band_path(car, "NE_SW", False)
        half = float(band.rsplit(":", 1)[1])
        self.assertLess(half, 0.35 + 0.35)   # body + lateral drift allowance only
        self.assertGreater(float(car_band.rsplit(":", 1)[1]), half + 0.85 - 0.35)
        self.sim.vehicles.pop()
        self.sim.vehicles.pop()


class GreenStandingTests(unittest.TestCase):
    def test_few_vehicles_stand_at_the_line_on_green(self):
        """Seed 2, 600 s: 1,022 vehicle-seconds before the rework (882 of
        them behind the keep-clear rule, 79 behind a false seam leader),
        156 after it (143 and 9)."""
        sim, held = green_standing(2, 3000)
        self.assertEqual(sim.jam_removed, 0)
        self.assertGreater(sim.passed, 1450)
        self.assertLess(held["total"], 400.0)
        self.assertLess(held["seam"], 40.0)


if __name__ == "__main__":
    unittest.main()
