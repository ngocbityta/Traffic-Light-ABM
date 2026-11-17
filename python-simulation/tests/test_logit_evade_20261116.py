"""Logit gap acceptance (Vu & Shimizu 2010) and lateral evasion in the
junction (cut tail / run in front), 16/11/2026."""
import math
import unittest

try:
    from python_simulation.junction_negotiation import (
        Agent, JunctionNegotiator, NegotiationParams)
    from python_simulation.python_sim import LeVanLuongSimulator
except ImportError:  # flat layout
    from junction_negotiation import Agent, JunctionNegotiator, NegotiationParams
    from python_sim import LeVanLuongSimulator

from test_junction_negotiation import agent, zmap


class LogitModelTests(unittest.TestCase):
    def setUp(self):
        self.neg = JunctionNegotiator(zmap(), NegotiationParams())

    def test_default_is_logit(self):
        self.assertEqual(NegotiationParams().gap_model, "logit")

    def test_motorcycle_coefficients(self):
        m = agent("m", "S_W@0:1", 8.0, 0.0, "left", "S", wait_time=0.0)
        m.group_moto = 1
        # V = 2.449 TG + 0.198 - 2.257 = 0 at TG = 0.8408 s
        self.assertAlmostEqual(self.neg.accept_probability(m, 2.059 / 2.449), 0.5, places=3)
        self.assertLess(self.neg.accept_probability(m, 0.0), 0.15)
        self.assertGreater(self.neg.accept_probability(m, 3.0), 0.99)
        # Waiting lowers, a larger motorcycle group raises acceptance.
        p0 = self.neg.accept_probability(m, 1.0)
        m.wait_time = 10.0
        self.assertLess(self.neg.accept_probability(m, 1.0), p0)
        m.wait_time, m.group_moto = 0.0, 4
        self.assertGreater(self.neg.accept_probability(m, 1.0), p0)

    def test_car_coefficients(self):
        c = agent("c", "S_W@0:1", 8.0, 0.0, "left", "S", motorbike=False, wait_time=0.0)
        v = 2.887 * 1.2 - 4.225
        self.assertAlmostEqual(self.neg.accept_probability(c, 1.2),
                               1.0 / (1.0 + math.exp(-v)), places=9)
        # Cars need a longer gap than motorcycles for the same probability.
        m = agent("m", "S_W@0:1", 8.0, 0.0, "left", "S", wait_time=0.0)
        self.assertLess(self.neg.accept_probability(c, 1.0),
                        self.neg.accept_probability(m, 1.0))

    def test_draw_is_deterministic_and_seeded(self):
        a = self.neg._draw("12", "34")
        self.assertEqual(a, self.neg._draw("12", "34"))
        self.assertTrue(0.0 <= a < 1.0)
        self.neg.seed = 5
        self.assertNotEqual(a, self.neg._draw("12", "34"))

    def test_conceded_minor_gives_way(self):
        m = agent("m", "S_W@0:1", 6.0, 3.0, "left", "S")
        f = agent("f", "N_S@0:1", 12.0, 6.0, "through", "N")
        z = self.neg.zones.zones(m.movement, f.movement)[0]
        self.neg._conceded.add(("m", "f"))
        self.assertEqual(self.neg.zone_verdict(m, f, z, None, {}), ("block", "conceded"))


class LogitSimulationTests(unittest.TestCase):
    def test_acceptance_curve_follows_the_logit(self):
        sim = LeVanLuongSimulator(seed=1)
        sim.step(1500)
        events = sim._negotiation.negotiator.gap_events
        self.assertGreater(len(events), 150)
        low = [e for e in events if e[2] < 0.0]
        high = [e for e in events if e[2] >= 2.0]
        self.assertTrue(low and high)
        self.assertLess(sum(e[4] for e in low) / len(low), 0.25)
        self.assertGreater(sum(e[4] for e in high) / len(high), 0.9)
        # Realised acceptance matches the model probability on average.
        mean_p = sum(e[3] for e in events) / len(events)
        mean_acc = sum(e[4] for e in events) / len(events)
        self.assertAlmostEqual(mean_acc, mean_p, delta=0.06)


class EvasionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = LeVanLuongSimulator(seed=2)
        cls.sim.step(2500)

    def test_both_strategies_occur_and_shares_are_plausible(self):
        ad = self.sim._negotiation
        kinds = {(e[2], e[3]) for e in ad.evade_events}
        self.assertIn(("cut_tail", True), kinds)
        self.assertIn(("run_in_front", True), kinds)
        moto = ad.evade_gap_events["moto"]
        cut = sum(1 for e in ad.evade_events if e[3] and e[2] == "cut_tail")
        self.assertGreater(moto, 80)
        self.assertTrue(0.08 < cut / moto < 0.35, cut / moto)

    def test_moves_stay_on_the_connector_and_nobody_jams(self):
        ad = self.sim._negotiation
        self.assertTrue(all(1 <= e[4] <= 3 for e in ad.evade_events))
        self.assertEqual(self.sim.jam_removed, 0)
        self.assertEqual(self.sim.road_safety_corrections, 0)
        for v in self.sim.vehicles:
            if v.road == v.conn_road:
                n = self.sim.roads[v.road]["num_lanes"]
                self.assertTrue(0 <= v.lane <= n - v.occupied)

    def test_disabled_evasion_does_nothing(self):
        sim = LeVanLuongSimulator(seed=2)
        sim._negotiation.negotiator.p.evade = False
        sim.step(600)
        self.assertEqual(sim._negotiation.evade_events, [])
        self.assertFalse(any(v.evading for v in sim.vehicles))


if __name__ == "__main__":
    unittest.main()
