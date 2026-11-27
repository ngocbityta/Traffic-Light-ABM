"""Extension control: +5 s per decision, no extension once the red countdown
is at 8 s or less (27/11/2026).

The rules under test (``signal_control="extend"``, train mode):
* every green starts with ``min_green`` planned and lasts min + k * ext_step;
* the policy is asked only when the red countdown of the waiting axis is
  ``countdown_lock + decision_lead`` = 9 s (5 s of green left with a 4 s
  yellow and no all-red);
* choosing the running phase adds 5 s, the other phase fixes the switch;
* a fixed switch never moves: the shown red countdown falls by DT each tick
  and the other axis gets green exactly when it reaches 0;
* at ``max_green`` the green is locked without asking;
* ``next_decision_in_cycles`` lands exactly on the next decision.
"""
import random
import tempfile
import unittest
from pathlib import Path

from python_simulation.demand import axis_flows, generate_demand
from python_simulation.python_sim import DT, LeVanLuongSimulator

HORIZON = 1500
_DEMAND = {}


def demand(seed, scale):
    key = (seed, scale)
    if key not in _DEMAND:
        path = Path(tempfile.mkdtemp()) / f"demand_{seed}_{scale}.csv"
        generate_demand(path, seed, HORIZON, heavy_ratio=.03,
                        flow_by_arm=axis_flows(scale, .1))
        _DEMAND[key] = str(path)
    return _DEMAND[key]


def make(seed=5, scale=1.0, **kwargs):
    params = dict(seed=seed, demand_file=demand(seed, scale), train_mode=True,
                  mode="safe", yellow=4.0, all_red=0.0, min_green=15.0,
                  signal_control="extend", max_green=75.0, observation_zone=50.0)
    params.update(kwargs)
    return LeVanLuongSimulator(**params)


def run(sim, policy, horizon=HORIZON, check=None):
    """Step one tick at a time so every rule is checked on every tick.

    ``next_decision_in_cycles`` is what the environment steps at once, so it
    must never jump over a decision: for every tick t before the next
    decision d, d >= t + hint(t). Right after a decision (or at reset) the
    hint is exact unless a green is locked automatically at max green.
    """
    decisions = []
    bound = sim.cycle + sim.next_decision_in_cycles
    exact = bound
    locks = sim.auto_locks
    while sim.cycle < horizon:
        if sim.dqn_waiting_action:
            assert sim.cycle >= bound, (sim.cycle, bound)
            if sim.auto_locks == locks:
                assert sim.cycle == exact, (sim.cycle, exact)
            action = policy(sim)
            decisions.append((sim.cycle, sim.phase, sim.green_ticks,
                              sim.planned_green_ticks, sim.red_countdown(), action))
            sim.set_action(action)
            sim.step(1)
            bound = exact = sim.cycle + sim.next_decision_in_cycles
            locks = sim.auto_locks
        else:
            sim.step(1)
            if not sim.dqn_waiting_action:
                bound = max(bound, sim.cycle + sim.next_decision_in_cycles)
        if check is not None:
            check(sim)
    return decisions


def extend_always(sim):
    return 0 if sim.phase == "LVL" else 2


def end_always(sim):
    return 2 if sim.phase == "LVL" else 0


class ExtensionControlTests(unittest.TestCase):
    def test_greens_are_min_plus_whole_extensions_and_within_max(self):
        rng = random.Random(3)
        sim = make()
        run(sim, lambda s: rng.choice((0, 2)))
        self.assertGreater(len(sim.green_log), 10)
        for phase, seconds, extensions in sim.green_log:
            self.assertAlmostEqual(seconds, 15.0 + 5.0 * extensions, places=6)
            self.assertLessEqual(seconds, 75.0 + 1e-9)
        self.assertEqual(sim.unanswered_decisions, 0)
        self.assertEqual(sim.jam_removed, 0)

    def test_decision_only_at_nine_seconds_of_red_left(self):
        rng = random.Random(4)
        sim = make()
        decisions = run(sim, lambda s: rng.choice((0, 2)))
        self.assertTrue(decisions)
        for cycle, phase, green, planned, red, action in decisions:
            self.assertIn(phase, ("LVL", "HDT"))
            self.assertEqual(planned - green, 25)           # 5 s of green left
            self.assertIsNone(red)                          # not fixed yet: hidden
        # The first decision of a green comes after 10 s of green.
        firsts = [d for d in decisions if d[3] == 75]
        self.assertTrue(all(d[2] == 50 for d in firsts))

    def test_fixed_switch_never_moves_and_countdown_is_truthful(self):
        rng = random.Random(5)
        sim = make(seed=6, scale=1.2)
        state = {"left": None, "shown": None, "switches": 0}

        def check(s):
            left = s.red_countdown()
            prev = state["left"]
            if prev is not None:
                if prev > 1e-9:
                    # A fixed end of green never moves: one tick less per tick.
                    self.assertIsNotNone(left)
                    self.assertAlmostEqual(left, prev - DT, places=6)
                else:
                    # It reached zero: the waiting axis has green now.
                    self.assertIn(s.phase, ("LVL", "HDT"))
                    self.assertFalse(s.signal_locked)
                    state["switches"] += 1
            shown = s.red_countdown_display()
            if shown is not None:
                self.assertLessEqual(shown, 8)
                if state["shown"] is not None:
                    self.assertLessEqual(shown, state["shown"])
            state["shown"] = shown
            # ``left`` counts red ticks after this one: 0 on the last yellow
            # (or all-red) tick, green on the next.
            state["left"] = None if (prev is not None and prev <= 1e-9) else left

        run(sim, lambda s: rng.choice((0, 2)), check=check)
        self.assertGreater(state["switches"], 5)

    def test_green_countdown_known_only_once_locked(self):
        sim = make()
        seen_unknown = seen_known = False
        while sim.cycle < 600:
            if sim.dqn_waiting_action:
                self.assertIsNone(sim._green_remaining())
                sim.set_action(end_always(sim))
            sim.step(1)
            left = sim._green_remaining()
            if sim.phase in ("LVL", "HDT"):
                if sim.signal_locked:
                    seen_known = True
                    self.assertIsNotNone(left)
                    self.assertLessEqual(left, 5.0 + 1e-9)
                else:
                    seen_unknown = True
                    self.assertIsNone(left)
        self.assertTrue(seen_known and seen_unknown)

    def test_always_extend_reaches_max_green_without_a_forced_decision(self):
        sim = make(seed=7, scale=1.3)
        decisions = run(sim, extend_always, horizon=1200)
        greens = [g for _, g, _ in sim.green_log]
        self.assertTrue(greens)
        self.assertTrue(all(abs(g - 75.0) < 1e-9 for g in greens))
        self.assertGreater(sim.auto_locks, 0)
        # 15 s -> 75 s takes 12 extensions, each one asked.
        per_green = [d for d in decisions if d[1] == "LVL"]
        self.assertGreaterEqual(len(per_green), 12)
        self.assertTrue(all(d[3] + 25 <= 375 for d in decisions))

    def test_always_end_gives_min_green_and_alternates(self):
        sim = make()
        run(sim, end_always, horizon=1200)
        phases = [p for p, _, _ in sim.green_log]
        self.assertTrue(all(abs(g - 15.0) < 1e-9 for _, g, _ in sim.green_log))
        self.assertTrue(all(a != b for a, b in zip(phases, phases[1:])))
        self.assertEqual(sim.last_green_seconds["LVL"], 15.0)

    def test_unanswered_decision_fixes_the_switch(self):
        sim = make()
        while not sim.dqn_waiting_action:
            sim.step(1)
        sim.step(1)                       # no set_action
        self.assertEqual(sim.unanswered_decisions, 1)
        self.assertTrue(sim.signal_locked)
        self.assertAlmostEqual(sim.red_countdown(), 9.0 - DT, places=6)

    def test_bad_timings_are_rejected(self):
        with self.assertRaises(ValueError):
            make(min_green=5.0)           # first decision would precede green
        with self.assertRaises(ValueError):
            make(max_green=10.0)
        with self.assertRaises(ValueError):
            make(countdown_lock=3.0, decision_lead=1.0)   # below the yellow
        with self.assertRaises(ValueError):
            make(signal_control="cycle")

    def test_all_red_shifts_the_decision_to_keep_the_lock(self):
        sim = make(all_red=2.0)
        decisions = run(sim, end_always, horizon=800)
        for _, _, green, planned, _, _ in decisions:
            self.assertEqual(planned - green, 15)           # 3 s of green left

    def test_zone_counters_are_consistent(self):
        sim = make(scale=.9)
        run(sim, end_always, horizon=HORIZON)
        for arm in sim.zone_arrivals:
            self.assertGreater(sim.zone_arrivals[arm], 0)
            # Every vehicle that crossed the line entered the zone first.
            self.assertGreaterEqual(sim.zone_arrivals[arm], sim.stopline_crossings[arm])
        self.assertTrue(all(t >= 0 for t in sim.zone_stop_time.values()))

    def test_zone_off_by_default(self):
        sim = make(observation_zone=0.0)
        sim.step(300)
        self.assertEqual(sum(sim.zone_arrivals.values()), 0)
        self.assertGreater(sum(sim.stopline_crossings.values()), 0)

    def test_deterministic(self):
        def trace():
            rng = random.Random(9)
            sim = make(seed=8)
            run(sim, lambda s: rng.choice((0, 2)), horizon=900)
            return (sim.green_log, sim.passed,
                    round(sim.weighted_delay_integral, 9), dict(sim.zone_arrivals))
        self.assertEqual(trace(), trace())


if __name__ == "__main__":
    unittest.main()
