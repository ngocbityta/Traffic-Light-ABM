"""CLI validation and saved metadata for the LVL Python simulation mode."""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import parse_args, training_config


class TrainPythonSimModeTests(unittest.TestCase):
    def test_default_is_safe_and_saved_in_training_config(self):
        args = parse_args(["--total-steps", "10"])
        self.assertEqual(args.python_sim_mode, "safe")
        self.assertEqual(training_config(args)["python_sim_mode"], "safe")

    def test_python_lvl_accepts_and_records_gama_parity_mode(self):
        args = parse_args(["--backend", "python", "--scenario", "lvl",
                           "--python-sim-mode", "gama_2025_6_4",
                           "--total-steps", "10"])
        self.assertEqual(training_config(args)["python_sim_mode"], "gama_2025_6_4")

    def test_gama_backend_keeps_safe_metadata_and_ignores_python_mode(self):
        args = parse_args(["--backend", "gama", "--total-steps", "10"])
        self.assertEqual(training_config(args)["python_sim_mode"], "safe")

    def test_lvl_training_fingerprint_covers_imported_python_sim_helpers(self):
        config = training_config(parse_args(["--backend", "python", "--scenario", "lvl",
                                            "--total-steps", "10"]))
        for source in ("python-simulation/simulation/python_simulation/gama_2025_idm_mobil.py",
                       "python-simulation/simulation/python_simulation/gama_2025_transition.py",
                       "python-simulation/simulation/python_simulation/safe_startup.py",
                       "python-simulation/simulation/python_simulation/vehicle_geometry.py",
                       "python-simulation/simulation/python_simulation/junction_conflicts.py",
                       "python-simulation/simulation/python_simulation/safe_collision_guard.py"):
            with self.subTest(source=source):
                self.assertRegex(config["code_sha256"][source], r"^[0-9a-f]{64}$")

    def test_nondefault_mode_is_rejected_when_backend_cannot_use_it(self):
        for backend_args in (("--backend", "gama"),
                             ("--backend", "python", "--scenario", "legacy")):
            with self.subTest(backend_args=backend_args):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    parse_args([*backend_args, "--python-sim-mode", "gama_2025_6_4",
                                "--total-steps", "10"])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("requires --backend python --scenario lvl", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
