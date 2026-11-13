import math
import unittest

import numpy as np

try:
    from python_simulation.junction_negotiation import (
        Agent, JunctionNegotiator, NegotiationParams, PathSample, Zone, ZoneMap,
        braking_arrival, travel_time)
except ImportError:  # flat layout
    from junction_negotiation import (
        Agent, JunctionNegotiator, NegotiationParams, PathSample, Zone, ZoneMap,
        braking_arrival, travel_time)


def line(p0, p1, half_width=0.35, step=0.25):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    s = np.linspace(0.0, length, int(length / step) + 1)
    pts = p0 + (p1 - p0)[None, :] * (s / length)[:, None]
    return PathSample(pts, s, half_width, length)


# Two-phase toy junction: "N" southbound through (major) and "S" northbound
# left turn approximated by a straight path crossing it at right angle.
PATHS = {
    "N_S@0:1": line((0.0, 20.0), (0.0, -20.0)),            # through, x = 0
    "N_S@3:1": line((3.0, 20.0), (3.0, -20.0)),            # through, x = 3 (other sublane)
    "S_W@0:1": line((10.0, 0.0), (-10.0, 0.0)),            # left turner crossing y = 0
    "S_W@1:1": line((10.0, 2.0), (-10.0, 2.0)),
}


def zmap():
    return ZoneMap(lambda m: PATHS[m], lateral_margin=0.2)


def agent(vid, movement, front, speed, turn, arm, **kw):
    base = dict(vid=vid, movement=movement, arm=arm, turn=turn, front=front,
                length=1.9, speed=speed, accel=2.0, decel=3.0, max_decel=6.0,
                min_gap=0.5, desired=11.0, motorbike=True, on_junction=front > 0,
                arm_green=True, road="conn" if front > 0 else "in")
    base.update(kw)
    return Agent(**base)


class ZoneTests(unittest.TestCase):
    def test_perpendicular_crossing_zone(self):
        zones = zmap().zones("N_S@0:1", "S_W@0:1")
        self.assertEqual(len(zones), 1)
        z = zones[0]
        self.assertEqual(z.kind, "cross")
        # Crossing at s = 20 on the through path and s = 10 on the left path;
        # half extent = 0.35 + 0.35 + 0.2 = 0.9 m.
        self.assertAlmostEqual((z.a_in + z.a_out) / 2, 20.0, delta=0.3)
        self.assertAlmostEqual((z.b_in + z.b_out) / 2, 10.0, delta=0.3)
        self.assertAlmostEqual(z.a_out - z.a_in, 1.75, delta=0.5)

    def test_swapped_view(self):
        m = zmap()
        a = m.zones("N_S@0:1", "S_W@0:1")[0]
        b = m.zones("S_W@0:1", "N_S@0:1")[0]
        self.assertEqual(a.swapped(), b)

    def test_parallel_separated_paths_do_not_conflict(self):
        # 3 m apart, two motorbikes: no conflict -> filtering past is free.
        self.assertEqual(zmap().zones("N_S@0:1", "N_S@3:1"), ())


class KinematicsTests(unittest.TestCase):
    def test_travel_time(self):
        self.assertAlmostEqual(travel_time(10.0, 0.0, 2.0, 100.0), math.sqrt(10.0), places=6)
        self.assertAlmostEqual(travel_time(10.0, 5.0, 0.0, 5.0), 2.0)

    def test_braking_arrival(self):
        self.assertEqual(braking_arrival(10.0, 5.0, 3.0), math.inf)   # stops in 4.2 m
        self.assertLess(braking_arrival(3.0, 10.0, 3.0), 0.4)


class MinorGateTests(unittest.TestCase):
    def setUp(self):
        self.neg = JunctionNegotiator(zmap(), NegotiationParams())

    def foes(self, minor, major):
        return [(major, self.neg.zones.zones(minor.movement, major.movement))]

    def test_far_major_leaves_gap(self):
        m = agent("m", "S_W@0:1", 8.0, 0.0, "left", "S")
        f = agent("f", "N_S@0:1", -60.0, 8.0, "through", "N")
        self.assertEqual(self.neg.minor_blocked(m, self.foes(m, f), False), (None, "gap"))

    def test_close_fast_major_blocks(self):
        m = agent("m", "S_W@0:1", 8.0, 0.0, "left", "S")
        f = agent("f", "N_S@0:1", 12.0, 10.0, "through", "N")  # cannot stop
        self.assertEqual(self.neg.minor_blocked(m, self.foes(m, f), False)[0], "f")

    def test_impatience_forces_a_major_that_can_brake(self):
        params = NegotiationParams(impatience_base_moto=0.6)
        neg = JunctionNegotiator(zmap(), params)
        m = agent("m", "S_W@0:1", 8.5, 0.0, "left", "S")
        f = agent("f", "N_S@0:1", -2.0, 6.0, "through", "N")   # 22 m away at 6 m/s
        calm = JunctionNegotiator(zmap(), NegotiationParams(impatience_base_moto=0.0))
        self.assertEqual(calm.minor_blocked(m, [(f, calm.zones.zones(m.movement, f.movement))], False)[0], "f")
        self.assertEqual(neg.minor_blocked(m, [(f, neg.zones.zones(m.movement, f.movement))], False),
                         (None, "forced"))

    def test_major_patience_stops_forcing(self):
        params = NegotiationParams(impatience_base_moto=0.6, max_yield_moto=3.0)
        neg = JunctionNegotiator(zmap(), params)
        m = agent("m", "S_W@0:1", 8.5, 0.0, "left", "S")
        f = agent("f", "N_S@0:1", 17.0, 0.0, "through", "N", hold_time=5.0)
        self.assertEqual(neg.minor_blocked(m, [(f, neg.zones.zones(m.movement, f.movement))], False)[0], "f")

    def test_group_follower_joins(self):
        m = agent("m", "S_W@0:1", 8.5, 1.0, "left", "S")
        f = agent("f", "N_S@0:1", 2.0, 6.0, "through", "N")
        self.assertEqual(self.neg.minor_blocked(m, self.foes(m, f), True), (None, "group"))

    def test_foe_in_zone_blocks(self):
        m = agent("m", "S_W@0:1", 8.0, 0.0, "left", "S")
        f = agent("f", "N_S@0:1", 20.5, 0.0, "through", "N")
        self.assertEqual(self.neg.minor_blocked(m, self.foes(m, f), True)[0], "f")


class DecideTests(unittest.TestCase):
    def test_green_waiter_ages_ahead_of_conflicting_uncommitted_traffic(self):
        neg = JunctionNegotiator(zmap())
        waiting = agent("waiting", "S_W@0:1", 2.0, 0.0, "left", "S",
                        wait_time=4.0)
        arriving = agent("arriving", "N_S@0:1", 17.5, 0.0, "through", "N")
        z = neg.zones.zones(waiting.movement, arriving.movement)[0]
        first, second, reason = neg.order_active(waiting, arriving, z)
        self.assertIs(first, waiting)
        self.assertIs(second, arriving)
        self.assertEqual(reason, "aged_priority")

    def test_aged_green_left_turn_uses_a_safe_gap_or_forces_a_brake(self):
        neg = JunctionNegotiator(zmap(), NegotiationParams(impatience_base_moto=0.0))
        waiting = agent("waiting", "S_W@0:1", 8.5, 0.0, "left", "S",
                        wait_time=4.0)
        foe = agent("foe", "N_S@0:1", -2.0, 6.0, "through", "N")
        foes = [(foe, neg.zones.zones(waiting.movement, foe.movement))]
        self.assertEqual(neg.minor_blocked(waiting, foes, False),
                         (None, "forced"))
        foe.speed = 11.0
        foe.front = 16.0
        self.assertEqual(neg.minor_blocked(waiting, foes, False)[0],
                         "foe")

    def test_minor_creeps_into_stream_zone_by_zone(self):
        # The far sublane (x=3) is takeable, the near one (x=0) is not: the
        # left turner inches past x=3 and stops inside the stream before x=0.
        neg = JunctionNegotiator(zmap(), NegotiationParams(impatience_base_moto=0.0))
        m = agent("m", "S_W@1:1", 5.0, 3.0, "left", "S")      # crosses x=3 then x=0
        f1 = agent("f1", "N_S@0:1", 6.0, 9.0, "through", "N")   # can stop, arrives in ~1.3 s
        f2 = agent("f2", "N_S@3:1", -60.0, 9.0, "through", "N")
        stop, _ = neg.decide([m, f1, f2], now=0.0)
        z_far = neg.zones.zones("S_W@1:1", "N_S@3:1")[0]
        z_near = neg.zones.zones("S_W@1:1", "N_S@0:1")[0]
        self.assertLess(z_far.a_in, z_near.a_in)
        self.assertAlmostEqual(stop["m"][0], z_near.a_in - m.min_gap, places=6)
        self.assertGreater(stop["m"][0], z_far.a_out)
        self.assertEqual(stop["f1"][0], math.inf)
        self.assertEqual(stop["f2"][0], math.inf)

    def test_forced_zone_makes_the_major_brake(self):
        neg = JunctionNegotiator(zmap(), NegotiationParams(impatience_base_moto=0.6))
        m = agent("m", "S_W@0:1", 8.5, 0.0, "left", "S")
        f = agent("f", "N_S@0:1", -2.0, 6.0, "through", "N")
        stop, _ = neg.decide([m, f], now=0.0)
        z = neg.zones.zones("N_S@0:1", "S_W@0:1")[0]
        self.assertEqual(stop["m"][0], math.inf)
        self.assertAlmostEqual(stop["f"][0], z.a_in - f.min_gap, places=6)

    def test_held_major_reclaims_after_patience(self):
        neg = JunctionNegotiator(zmap(), NegotiationParams(impatience_base_moto=0.6))
        m = agent("m", "S_W@0:1", 8.5, 0.0, "left", "S")
        f = agent("f", "N_S@0:1", 17.0, 0.0, "through", "N", hold_time=2.5)
        z = neg.zones.zones("S_W@0:1", "N_S@0:1")[0]
        self.assertEqual(neg.zone_verdict(m, f, z, None, {}), ("block", "reclaim"))
        f.hold_time = 0.4
        self.assertEqual(neg.zone_verdict(m, f, z, None, {}), ("ok", "forced"))

    def test_through_at_stop_line_is_not_forced_by_a_later_left(self):
        # SUMO: impatience only applies when the minor would arrive first.
        neg = JunctionNegotiator(zmap(), NegotiationParams(impatience_base_moto=0.9))
        m = agent("m", "S_W@0:1", 2.0, 0.0, "left", "S")        # 7 m from the zone
        f = agent("f", "N_S@0:1", 17.5, 0.0, "through", "N")    # 1.6 m from the zone
        z = neg.zones.zones("S_W@0:1", "N_S@0:1")[0]
        self.assertEqual(neg.zone_verdict(m, f, z, None, {}), ("block", "foe_first"))

    def test_new_phase_waits_before_zone_of_clearing_vehicle(self):
        neg = JunctionNegotiator(zmap())
        c = agent("c", "S_W@0:1", 6.0, 0.0, "left", "S", arm_green=False)
        f = agent("f", "N_S@0:1", 2.0, 4.0, "through", "N")
        stop, _ = neg.decide([c, f], now=0.0)
        z = neg.zones.zones("N_S@0:1", "S_W@0:1")[0]
        self.assertAlmostEqual(stop["f"][0], z.a_in - f.min_gap, places=6)
        self.assertEqual(stop["c"][0], math.inf)

    def test_committed_minor_makes_major_yield(self):
        neg = JunctionNegotiator(zmap())
        m = agent("m", "S_W@0:1", 9.0, 4.0, "left", "S", commit_time=0.0)
        f = agent("f", "N_S@0:1", 15.0, 5.0, "through", "N")   # could arrive before m clears
        stop, _ = neg.decide([m, f], now=0.2)
        z = neg.zones.zones("N_S@0:1", "S_W@0:1")[0]
        self.assertAlmostEqual(stop["f"][0], z.a_in - f.min_gap, places=6)
        self.assertEqual(stop["m"][0], math.inf)

    def test_major_that_cannot_stop_keeps_priority(self):
        neg = JunctionNegotiator(zmap())
        m = agent("m", "S_W@0:1", 7.0, 2.0, "left", "S", commit_time=0.0)
        f = agent("f", "N_S@0:1", 16.0, 11.0, "through", "N")
        stop, _ = neg.decide([m, f], now=0.2)
        self.assertEqual(stop["f"][0], math.inf)
        self.assertLess(stop["m"][0], 10.0)

    def test_major_far_behind_is_not_slowed(self):
        # jmAdvance: a major that cannot reach the zone before the committed
        # minor clears it gets no stop target.
        neg = JunctionNegotiator(zmap())
        m = agent("m", "S_W@0:1", 9.0, 4.0, "left", "S", commit_time=0.0)
        f = agent("f", "N_S@0:1", 2.0, 5.0, "through", "N")
        stop, _ = neg.decide([m, f], now=0.2)
        self.assertEqual(stop["f"][0], math.inf)

    def test_minor_free_when_clear(self):
        neg = JunctionNegotiator(zmap())
        m = agent("m", "S_W@0:1", 8.0, 1.0, "left", "S")
        f = agent("f", "N_S@0:1", -80.0, 8.0, "through", "N")
        stop, commits = neg.decide([m, f], now=1.0)
        self.assertEqual(stop["m"][0], math.inf)

    def test_cycle_breaker_flips_non_physical_edge(self):
        neg = JunctionNegotiator(zmap())
        a = agent("a", "N_S@0:1", 0.0, 0.0, "through", "N", hold_time=4.0)
        b = agent("b", "S_W@0:1", 0.0, 0.0, "left", "S")
        z = Zone(1.0, 2.0, 1.0, 2.0, "cross")
        edges = [[a, b, [z], "committed"], [b, a, [z], "mutual"]]
        neg._break_cycles(edges)
        self.assertIn("deadlock_break", [e[3] for e in edges])


if __name__ == "__main__":
    unittest.main()


class SimulatorIntegrationTests(unittest.TestCase):
    def _sim(self, **kw):
        try:
            from python_simulation.python_sim import LeVanLuongSimulator
        except ImportError:
            from python_sim import LeVanLuongSimulator
        return LeVanLuongSimulator(**kw)

    def test_default_is_negotiation(self):
        sim = self._sim(seed=1)
        self.assertEqual(sim.junction_model, "negotiation")
        self.assertIsNotNone(sim._negotiation)

    def test_negotiation_runs_and_discharges(self):
        sim = self._sim(seed=1, junction_model="negotiation", flow_per_approach=1500.0)
        sim.step(400)
        self.assertGreater(sim.passed, 20)
        stats = sim._negotiation.negotiator.stats
        self.assertEqual(stats["both_in_zone"], 0)

    def test_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            self._sim(seed=1, junction_model="magic")
