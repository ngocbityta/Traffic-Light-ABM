"""Python training adapter passes the requested backend mode into each episode."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lvl_env import LVLPythonEnv


class LVLPythonEnvModeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.demand = Path(self.tmp.name) / "paired.csv"
        # One NE motorbike at tick 1, lane 0, going through.
        self.demand.write_text("1,0,0,40,2,0\n", encoding="utf-8")
        self.parameters = [{"name": "demand_file_path", "type": "string",
                            "value": str(self.demand)}]

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def make_env(self, **kwargs):
        return LVLPythonEnv(state_mode="spatial", reward_mode="r1", **kwargs)

    def test_safe_mode_is_default_and_values_are_validated(self):
        self.assertEqual(self.make_env().python_sim_mode, "safe")
        with self.assertRaisesRegex(ValueError, "Unknown Python simulator mode"):
            self.make_env(python_sim_mode="gama")

    async def test_selected_mode_receives_paired_demand_and_drives_observations_and_reward(self):
        for mode in ("safe", "gama_2025_6_4"):
            with self.subTest(mode=mode):
                env = self.make_env(python_sim_mode=mode)
                initial = await env.reset(seed=17, parameters=self.parameters)
                self.assertEqual(env.sim.mode, mode)
                self.assertEqual(initial["cycle"], 0)
                self.assertEqual(initial["total_vehicles_created"], 0)

                await env.step_cycles(1)
                spawned = await env.read_state()
                self.assertEqual(spawned["total_vehicles_created"], 1)
                self.assertEqual(spawned["spatial"].shape, (4, 6, 20, 16))
                self.assertEqual(spawned["spatial"][0, 0, 19, 0], 1.0)

                await env.step_cycles(49)
                decision = await env.read_state()
                self.assertTrue(decision["dqn_waiting_action"])
                self.assertGreater(decision["system_time_integral"], 0.0)
                next_state, reward, truncated, info = await env.step(
                    0, horizon=150, current_state=decision)
                self.assertFalse(truncated)
                self.assertTrue(info["bootstrap_valid"])
                self.assertEqual(next_state["cycle"], 100)
                self.assertLess(reward, 0.0)
                self.assertAlmostEqual(
                    reward,
                    -(next_state["system_time_integral"] -
                      decision["system_time_integral"]) / env.r1_ref, places=6)

    async def test_pending_parity_spawn_is_absent_from_approach_grid(self):
        env = self.make_env(python_sim_mode="gama_2025_6_4")
        await env.reset(seed=17, parameters=self.parameters)
        env.sim._create_vehicle("NE", "through", True, 2, 40.0,
                                demand_row_id=9)
        state = await env.read_state()
        self.assertEqual(state["total_vehicles_created"], 1)
        self.assertEqual(state["approach_ns"], 0)
        self.assertEqual(state["spatial"][0, 0].sum(), 0.0)

    async def test_opposing_approach_is_observed_then_released_after_green_switch(self):
        env = self.make_env()
        demand = Path(self.tmp.name) / "opposing.csv"
        # SE is red during the initial LVL phase; action 2 gives it HDT green.
        demand.write_text("1,1,0,40,2,0\n", encoding="utf-8")
        parameters = [{"name": "demand_file_path", "type": "string",
                       "value": str(demand)}]
        await env.reset(seed=22, parameters=parameters)
        await env.step_cycles(50)
        before = await env.read_state()
        self.assertTrue(before["dqn_waiting_action"])
        self.assertEqual(before["global_phase"], 0)
        self.assertEqual(before["approach_ew"], 1)
        self.assertGreater(before["spatial"][1, 0].sum(), 0.0)

        after, reward, _, info = await env.step(2, horizon=150, current_state=before)
        self.assertEqual(after["global_phase"], 4)
        self.assertEqual(after["approach_ew"], 0)
        self.assertEqual(after["spatial"][1, 0].sum(), 0.0)
        self.assertEqual(env.sim.vehicles[0].road, "NW_out")
        self.assertTrue(info["bootstrap_valid"])
        self.assertLess(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
