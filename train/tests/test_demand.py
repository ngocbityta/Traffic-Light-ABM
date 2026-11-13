"""Checks that paired demand is deterministic and includes all routes."""

import csv
import tempfile
import unittest
from pathlib import Path

from train.logic.demand import generate_demand


class DemandTests(unittest.TestCase):
    def test_same_seed_same_requests_and_all_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = (Path(tmp) / name for name in ("a.csv", "b.csv", "c.csv"))
            same_a = generate_demand(a, 5001, 400)
            same_b = generate_demand(b, 5001, 400)
            different = generate_demand(c, 5002, 400)
            self.assertEqual(same_a, same_b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertNotEqual(same_a, different)
            with a.open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual({int(row[1]) for row in rows}, set(range(12)))
            self.assertEqual([int(row[0]) for row in rows],
                             sorted(int(row[0]) for row in rows))
            self.assertTrue(all(len(row) == 6 for row in rows))

    def test_lvl_site_uses_four_arms_and_assumed_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            self.assertEqual(generate_demand(a, 7, 8000, site="lvl"), generate_demand(b, 7, 8000, site="lvl"))
            with a.open(newline="") as stream:
                rows = [[float(x) for x in row] for row in csv.reader(stream)]
            self.assertEqual({int(r[1]) for r in rows}, set(range(4)))
            self.assertTrue({int(r[2]) for r in rows} <= {0, 1} and {int(r[4]) for r in rows} <= {0, 1, 2})
            per_arm = len(rows) / 4
            self.assertAlmostEqual(per_arm, 2400 * 800 / 3600, delta=0.15 * 2400 * 800 / 3600)
            self.assertAlmostEqual(sum(r[2] == 0 for r in rows) / len(rows), .85, delta=.04)


if __name__ == "__main__":
    unittest.main()
