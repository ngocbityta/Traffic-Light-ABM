"""CLI parity thresholds reject negative and non-finite limits."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_lvl_training_envs import validate_thresholds


def limits(**overrides):
    values = dict(max_created_delta=3, max_passed_delta=5,
                  max_queue_delta=10.0, max_reward_relative_error=0.1,
                  max_integral_relative_error=0.1, max_spatial_mae=0.05,
                  max_aux_mae=0.02, max_phase_mismatches=0,
                  max_action_mismatches=0)
    values.update(overrides)
    return SimpleNamespace(**values)


class CompareLVLThresholdTests(unittest.TestCase):
    def test_accepts_finite_nonnegative_thresholds(self):
        validate_thresholds(limits())

    def test_rejects_nonfinite_thresholds(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "finite and non-negative"):
                validate_thresholds(limits(max_spatial_mae=value))

    def test_rejects_negative_thresholds(self):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            validate_thresholds(limits(max_passed_delta=-1))


if __name__ == "__main__":
    unittest.main()
