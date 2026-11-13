"""Regression test for a yellow phase extended by junction clearance."""

import unittest
import numpy as np

from train.logic.gama_env import GamaTrafficEnv, GamaEnvError, STATE_PACK_EXPR, DTSE_PACK_EXPR


class GamaEnvScheduleTest(unittest.IsolatedAsyncioTestCase):
    async def test_large_cycle_request_uses_bounded_batches(self):
        env = GamaTrafficEnv(verbose=False)
        env.exp_id = "test"
        chunks = []
        class Client:
            async def step_awaitable(self, exp_id, count, **kwargs):
                chunks.append(count)
                return {"type": "CommandExecutedSuccessfully"}
        env.client = Client()
        await env.step_cycles(1100)
        self.assertEqual(chunks, [500, 500, 100])

    async def test_decision_snapshot_uses_one_packed_dtse_expression(self):
        env = GamaTrafficEnv(verbose=False)
        calls = []
        class Client:
            async def ask_awaitable(self, *args, **kwargs):
                calls.append("capture")
                return {"type": "CommandExecutedSuccessfully"}
        env.client = Client()
        env.exp_id = "test"
        async def expression(expr):
            calls.append(expr)
            if expr == STATE_PACK_EXPR:
                return "100|0|100|0|1|2|3|4|5|6|7|8|9|10|11|12|true|1"
            if expr == DTSE_PACK_EXPR:
                return "[" + ",".join("0.0" for _ in range(480)) + "]|2.5|1.5|4.0|1.25|[1,0,1,0]"
            raise AssertionError("Unexpected expression")
        env._exec_expression = expression
        state = await env.read_state()
        self.assertEqual(calls, [STATE_PACK_EXPR, "capture", DTSE_PACK_EXPR])
        self.assertEqual(state["dtse"].shape, (3, 8, 20))
        self.assertEqual(state["delay_sq"], 2.5)
        self.assertEqual(state["queue_time_integral"], 4.0)
        self.assertEqual(state["external_wait_integral"], 1.25)
        np.testing.assert_array_equal(state["action_mask"], [True, False, True, False])

    async def test_total_wait_reward_includes_external_queue(self):
        env = GamaTrafficEnv(verbose=False, reward_mode="total_wait")
        before = {"queue_time_integral": 5.0, "external_wait_integral": 2.0}
        after = {"queue_time_integral": 8.0, "external_wait_integral": 6.0}
        self.assertAlmostEqual(env.transition_reward(before, after), -7.0 / 50.0)

    async def test_paper_reward_uses_running_max_total_squared_delay(self):
        env = GamaTrafficEnv(verbose=False, reward_mode="reference")
        self.assertIsNone(env.max_delay_sq)
        self.assertEqual(env.reward_from_state({"delay_sq": 5.0}), 0.0)
        self.assertAlmostEqual(env.reward_from_state({"delay_sq": 2.0}), 0.6)
        self.assertEqual(env.reward_from_state({"delay_sq": 8.0}), 0.0)
        self.assertAlmostEqual(env.reward_from_state({"delay_sq": 4.0}), 0.5)

    async def test_reward_normalizer_survives_episode_reset(self):
        env = GamaTrafficEnv(verbose=False)
        env._worst_delay_sq = 10.0

        async def load_experiment(parameters=None):
            env.exp_id = "test"

        async def read_state():
            return {"cycle": 0}

        env.load_experiment = load_experiment
        env.read_state = read_state
        async def expression(expr):
            self.assertEqual(expr, "world.seed")
            return 17.0
        env._exec_expression = expression
        await env.reset(seed=17)
        self.assertEqual(env.reward_from_state({"delay_sq": 5.0}), 0.5)

    async def test_reset_rejects_a_seed_ignored_by_gama(self):
        env = GamaTrafficEnv(verbose=False)
        async def load_experiment(parameters=None):
            env.exp_id = "test"
        async def read_state():
            return {"cycle": 0}
        async def expression(expr):
            return 16.0
        env.load_experiment = load_experiment
        env.read_state = read_state
        env._exec_expression = expression
        with self.assertRaisesRegex(GamaEnvError, "applied seed 16"):
            await env.reset(seed=17)

    async def test_step_reaches_an_actual_decision_after_extended_yellow(self):
        env = GamaTrafficEnv(verbose=False)
        env._worst_delay_sq = 10.0
        states = iter([
            {"cycle": 100, "dqn_waiting_action": True},
            {"cycle": 381, "dqn_waiting_action": False, "next_decision_in_cycles": 1},
            {"cycle": 382, "dqn_waiting_action": True,
             "current_queue_length": 4, "average_waiting_time": 2.0,
             "delay_sq": 5.0},
        ])
        stepped = []
        actions = []

        async def read_state():
            return next(states)

        async def step_cycles(n):
            stepped.append(n)

        async def set_action(action):
            actions.append(action)

        async def next_decision():
            return 280

        env.read_state = read_state
        env.step_cycles = step_cycles
        env.set_action = set_action
        env.get_next_decision_in = next_decision

        state, reward, done, info = await env.step(1)
        self.assertEqual(actions, [1])
        self.assertEqual(stepped, [1, 280, 1])
        self.assertTrue(state["dqn_waiting_action"])
        self.assertEqual(reward, 0.5)
        self.assertFalse(done)
        self.assertEqual(info["next_decision_in_cycles"], 281)

    async def test_step_stops_at_horizon_during_clearance(self):
        env = GamaTrafficEnv(verbose=False)
        env._worst_delay_sq = 10.0
        final = {"cycle": 200, "dqn_waiting_action": False, "next_decision_in_cycles": 1,
                 "delay_sq": 5.0, "dtse": np.zeros((3,8,20))}
        states = iter([{"cycle": 100, "dqn_waiting_action": True},
                       {k:v for k,v in final.items() if k != "dtse"}, final])
        stepped = []
        async def read_state(**kwargs): return next(states)
        async def step_cycles(n): stepped.append(n)
        async def set_action(a): pass
        async def schedule(): return 150
        env.read_state, env.step_cycles = read_state, step_cycles
        env.set_action, env.get_next_decision_in = set_action, schedule
        state, reward, done, info = await env.step(2, horizon=200)
        self.assertEqual(stepped, [1, 99])
        self.assertEqual(state["cycle"], 200)
        self.assertTrue(done and info["truncated"])
        self.assertFalse(info["bootstrap_valid"])
        self.assertEqual(info["elapsed_seconds"], 10.)


if __name__ == "__main__":
    unittest.main()
