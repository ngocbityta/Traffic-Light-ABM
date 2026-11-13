import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("audit_lvl_discharge.py")
SPEC = importlib.util.spec_from_file_location("audit_lvl_discharge", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def state(identifier, arm, speed, lane=0):
    return {"id": identifier, "arm": arm, "lane": lane, "occupied": 1,
            "speed": speed, "road_state": (arm + "_in", 12.0)}


class DischargeTrackerTests(unittest.TestCase):
    def test_snapshots_onset_cohort_and_first_release(self):
        tracker = audit.DischargeTracker()
        before = [state("stopped", "SE", 0), state("moving", "SE", 0.2),
                  state("other_phase", "NE", 0)]
        after = [state("stopped", "SE", 0.11), state("moving", "SE", 0.3),
                 state("other_phase", "NE", 0.0)]
        tracker.observe(266, "none", "HDT", before, after)
        self.assertEqual(len(tracker.events), 1)
        cohort = tracker.events[0]["cohort"]
        self.assertEqual([v["id"] for v in cohort], ["stopped"])
        self.assertEqual(cohort[0]["release_tick"], 266)
        self.assertEqual(cohort[0]["release_time_sec"], 53.2)

    def test_no_cohort_for_non_green_approach_or_non_transition(self):
        tracker = audit.DischargeTracker()
        stopped = [state("a", "NE", 0)]
        tracker.observe(2, "LVL", "LVL", stopped, stopped)
        tracker.observe(3, "none", "HDT", stopped, stopped)
        self.assertEqual(len(tracker.events), 1)
        self.assertEqual(tracker.events[0]["cohort"], [])


if __name__ == "__main__":
    unittest.main()
