"""LVL observation fields must state the limited meaning of time metrics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lvl_env import _observation


class LVLMetricSemanticsTests(unittest.TestCase):
    def test_integral_ratios_have_explicit_names_and_keep_legacy_aliases(self):
        state = _observation(
            cycle=50, phase="LVL", waiting=True, next_decision=50,
            created=10, passed=4, queue_integral=25.0,
            system_integral=80.0, weighted_integral=0.0,
            vehicles=[], state_mode="spatial", green_cycles=50)
        self.assertEqual(state["cumulative_queue_time_per_created_vehicle"], 2.5)
        self.assertEqual(state["average_waiting_time"], 2.5)
        self.assertEqual(state["cumulative_system_time_per_passed_vehicle"], 20.0)
        self.assertEqual(state["average_travel_time"], 20.0)
        self.assertIn("not completed-vehicle mean", state["time_metric_semantics"])
        self.assertTrue(state["max_waiting_time_semantics"].startswith("unavailable"))
        self.assertNotEqual(state["max_waiting_time"], state["max_waiting_time"])


if __name__ == "__main__":
    unittest.main()
