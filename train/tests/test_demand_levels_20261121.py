"""Per-episode demand level: random scale and NW-SE axis imbalance (21/11/2026)."""
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from python_simulation.demand import (ARMS, BASE_FLOW, axis_flows, generate_demand,
                                      sample_demand_level)
from train.logic.train import parse_args, training_config

BASE = ["--scenario", "lvl", "--backend", "python", "--algorithm", "rainbow",
        "--reward-mode", "r1"]
RANDOM = BASE + ["--demand-scale-min", "0.5", "--demand-scale-max", "1.5",
                 "--axis-imbalance-max", "0.2"]


class AxisFlowsTest(unittest.TestCase):
    def test_mean_and_ratio(self):
        for scale in (0.5, 1.0, 1.5):
            for imbalance in (0.0, 0.1, 0.2):
                flows = axis_flows(scale, imbalance)
                self.assertAlmostEqual(sum(flows.values()) / 4, BASE_FLOW * scale)
                # Hoang Dao Thuy (NW-SE) is the busier axis.
                self.assertAlmostEqual(flows["NW"], flows["SE"])
                self.assertAlmostEqual(flows["NE"], flows["SW"])
                self.assertAlmostEqual(flows["NW"] / flows["NE"], 1 + imbalance)

    def test_default_level_keeps_the_old_demand_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = generate_demand(Path(tmp, "a.csv"), 1001, 3000)
            new = generate_demand(Path(tmp, "b.csv"), 1001, 3000,
                                  flow_by_arm=axis_flows(1.0, 0.0))
            self.assertEqual(old, new)

    def test_arrivals_follow_the_arm_flows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "d.csv")
            horizon = 40_000
            flows = axis_flows(1.5, 0.2)
            generate_demand(path, 5, horizon, flow_by_arm=flows)
            with open(path) as stream:
                rows = list(csv.reader(stream))[1:]
            counts = Counter(ARMS[int(row[1])] for row in rows)
            seconds = horizon * 0.2
            for arm in ARMS:
                self.assertAlmostEqual(counts[arm] / seconds * 3600 / flows[arm], 1.0, delta=0.04)

    def test_level_is_fixed_by_the_seed_and_spans_the_range(self):
        self.assertEqual(sample_demand_level(1234, 0.5, 1.5, 0.2),
                         sample_demand_level(1234, 0.5, 1.5, 0.2))
        levels = [sample_demand_level(seed, 0.5, 1.5, 0.2) for seed in range(1000, 3000)]
        scales = [s for s, _ in levels]
        imbalances = [i for _, i in levels]
        self.assertTrue(all(0.5 <= s <= 1.5 for s in scales))
        self.assertTrue(all(0.0 <= i <= 0.2 for i in imbalances))
        self.assertLess(min(scales), 0.55)
        self.assertGreater(max(scales), 1.45)
        self.assertAlmostEqual(sum(scales) / len(scales), 1.0, delta=0.03)
        self.assertEqual(sample_demand_level(7), (1.0, 0.0))


class TrainArgsTest(unittest.TestCase):
    def test_defaults_keep_the_fixed_demand(self):
        args = parse_args(BASE)
        self.assertEqual((args.demand_scale_min, args.demand_scale_max,
                          args.axis_imbalance_max), (1.0, 1.0, 0.0))
        self.assertEqual(args.validation_levels, ((1.0, 0.0), (1.0, 0.0)))

    def test_random_levels_validate_at_three_fixed_levels(self):
        args = parse_args(RANDOM)
        self.assertEqual(args.validation_levels, ((0.5, 0.1), (1.0, 0.1), (1.5, 0.1)))
        config = training_config(args)
        self.assertEqual(config["demand_scale_max"], 1.5)
        self.assertEqual(config["axis_imbalance_max"], 0.2)
        args = parse_args(BASE + ["--demand-scale-min", "0.5", "--demand-scale-max", "1.3",
                                  "--axis-imbalance-max", "0.2"])
        self.assertEqual(args.validation_levels, ((0.5, 0.1), (0.9, 0.1), (1.3, 0.1)))
        args = parse_args(RANDOM + ["--validation-levels", "0.7:0,1.3:0.2"])
        self.assertEqual(args.validation_levels, ((0.7, 0.0), (1.3, 0.2)))

    def test_bad_levels_are_rejected(self):
        for extra in (["--demand-scale-min", "1.6", "--demand-scale-max", "1.5"],
                      ["--demand-scale-min", "0"],
                      ["--axis-imbalance-max", "-0.1"],
                      ["--validation-levels", "1.0"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parse_args(BASE + extra)


class ResumeAndObservationTest(unittest.TestCase):
    def test_config_survives_the_json_round_trip(self):
        import json
        from train.logic.train import check_resume_config
        config = training_config(parse_args(RANDOM))
        self.assertEqual(check_resume_config(json.loads(json.dumps(config)), config), [])

    def test_log_queue_features_do_not_saturate(self):
        import numpy as np
        from train.logic.gama_env import featurize
        from train.logic.train import reward_settings

        def state(queue):
            return {"spatial": np.zeros((4, 6, 20, 16), np.float32), "global_phase": 0,
                    "green_cycles": 0, "queue_ns": queue, "queue_ew": 10,
                    "approach_ns": queue, "approach_ew": 10,
                    "pressure": np.asarray([queue, 0, 10, 0], np.float32),
                    "action_mask": np.array([True, False, True, False])}
        linear = [featurize(state(q))["aux"][5] for q in (150, 600)]
        self.assertEqual(linear[0], linear[1])          # both clipped at 1
        log = [featurize(state(q), "log")["aux"] for q in (0, 10, 150, 600)]
        self.assertEqual(log[0][5], 0.0)
        self.assertTrue(log[1][5] < log[2][5] < log[3][5] < 1.0)
        self.assertTrue(log[2][9] < log[3][9] < 1.0)
        self.assertEqual(reward_settings(parse_args(BASE)).get("queue_feature_scale"), None)
        self.assertEqual(reward_settings(parse_args(
            BASE + ["--queue-feature-scale", "log"]))["queue_feature_scale"], "log")


if __name__ == "__main__":
    unittest.main()
