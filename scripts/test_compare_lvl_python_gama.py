import unittest

from python_simulation.python_sim import LeVanLuongSimulator
from scripts.compare_lvl_python_gama import (
    compare_vehicle_states, parse_gama_vehicles, python_vehicle_states)


class CompareLvlPythonGamaTests(unittest.TestCase):
    def test_parse_gama_vehicle_state_and_position(self):
        states = parse_gama_vehicles("7:8.5:2:1:NE_in:12.0", {
            "NE_in": {"length": 100.0},
        })
        self.assertEqual(states[7]["road"], "NE_in")
        self.assertEqual(states[7]["lane"], 2)
        self.assertEqual(states[7]["position_m"], 88.0)

    def test_compare_reports_structural_and_numeric_divergence(self):
        gama = {
            1: {"road": "NE_in", "lane": 0, "speed_mps": 4.0,
                "position_m": 10.0},
            2: {"road": "SE_in", "lane": 0, "speed_mps": 4.0,
                "position_m": 10.0},
        }
        python = {
            1: {"road": "NE_in", "lane": 0, "speed_mps": 4.2,
                "position_m": 10.0},
            2: {"road": "SE_in", "lane": 1, "speed_mps": 4.0,
                "position_m": 10.0},
        }
        differences, first = compare_vehicle_states(
            gama, python, position_tolerance=0.05, speed_tolerance=0.05)
        self.assertEqual(first["kind"], "numeric")
        self.assertEqual(first["demand_row_id"], 1)
        self.assertEqual(differences[2]["kind"], "lane")

    def test_compare_reports_missing_vehicle(self):
        differences, first = compare_vehicle_states(
            {3: {"road": None, "lane": 0}}, {},
            position_tolerance=0.05, speed_tolerance=0.05)
        self.assertEqual(first["kind"], "presence")
        self.assertEqual(differences[3]["kind"], "presence")

    def test_pending_gama_spawn_has_nil_current_road(self):
        simulator = LeVanLuongSimulator(flow_per_approach=0,
                                       mode="gama_2025_6_4")
        simulator._create_vehicle("NE", "through", True, 2, 40.0,
                                  demand_row_id=7)
        state = python_vehicle_states(simulator)[7]
        self.assertIsNone(state["road"])
        self.assertIsNone(state["position_m"])
        self.assertEqual(state["speed_mps"], 0.0)


if __name__ == "__main__":
    unittest.main()
