"""Behavioral checks for the spatial Rainbow observation and learning path."""

import math
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from train.logic.rainbow_agent import RainbowAgent, pack_observation, unpack_observations
from train.logic.rainbow_agent import PREVIOUS_ARCHITECTURE
from train.logic.gama_env import GamaTrafficEnv, featurize
from train.logic.train import collect_rollout


def observation(lateral=0.0):
    grid = np.zeros((4, 6, 20, 16), dtype=np.float32)
    grid[:, 5] = 1.0
    grid[1, 0, 3, 5] = .75
    grid[1, 4, 3, 5] = lateral
    return {"grid": grid, "aux": np.arange(17, dtype=np.float32) / 17}


class RainbowTests(unittest.TestCase):
    def test_spatial_encoding_preserves_signed_lateral_motion(self):
        state = observation(-.6)
        blob, aux = pack_observation(state)
        grid, restored_aux = unpack_observations([blob], [aux])
        self.assertEqual(grid.shape, (1, 4, 6, 20, 16))
        self.assertAlmostEqual(float(grid[0, 1, 0, 3, 5]), .75, delta=.01)
        self.assertAlmostEqual(float(grid[0, 1, 4, 3, 5]), -.6, delta=.01)
        self.assertEqual(float(grid[0, 0, 4, 0, 0]), 0.0)
        np.testing.assert_allclose(restored_aux[0], state["aux"], atol=.001)

    def test_variable_duration_n_step_and_terminal_tail(self):
        agent = RainbowAgent(gamma=.9, n_step=3, buffer_size=8, batch_size=1,
                             warmup=0, device=torch.device("cpu"), seed=4)
        state = observation()
        agent.store(state, 0, -1, state, False, seconds=5, next_mask=[1, 1, 1, 1])
        agent.store(state, 0, -2, state, False, seconds=10, next_mask=[1, 1, 1, 1])
        agent.store(state, 0, -3, state, True, seconds=5, next_mask=[1, 1, 1, 1])
        self.assertEqual(len(agent.buffer), 3)
        self.assertAlmostEqual(float(agent.buffer.rewards[0]), -1 - .9 * 2 - .9 ** 3 * 3, places=5)
        self.assertAlmostEqual(float(agent.buffer.discounts[0]), .9 ** 4, places=5)
        self.assertTrue(agent.buffer.dones[:3].all())
        self.assertEqual(len(agent.pending), 0)

    def test_masked_action_update_and_training_restore(self):
        state = observation(.4)
        agent = RainbowAgent(buffer_size=8, batch_size=2, warmup=2,
                             n_step=2, device=torch.device("cpu"), seed=7)
        self.assertEqual(agent.act(state, greedy=True, action_mask=[0, 0, 1, 0]), 2)
        for i in range(4):
            agent.store(state, i % 4, -i - 1, state, i == 3,
                        seconds=5, next_mask=[1, 0, 1, 0])
        before = next(agent.online_net.parameters()).detach().clone()
        loss = agent.update()
        self.assertTrue(math.isfinite(loss))
        self.assertFalse(torch.equal(before, next(agent.online_net.parameters()).detach()))
        snapshot = agent.training_state()
        restored = RainbowAgent(buffer_size=8, batch_size=2, warmup=2,
                                n_step=2, device=torch.device("cpu"), seed=8)
        restored.restore_training(snapshot)
        self.assertEqual(len(restored.buffer), len(agent.buffer))
        self.assertEqual(restored.update_count, agent.update_count)

    def test_four_reward_integrals_and_auxiliary_phase(self):
        old = {"system_time_integral": 3, "stopped_time_integral": 2,
               "external_all_time_integral": 1, "outgoing_block_integral": 0,
               "weighted_delay_integral": 4}
        new = {"system_time_integral": 28, "stopped_time_integral": 12,
               "external_all_time_integral": 6, "outgoing_block_integral": 8,
               "weighted_delay_integral": 19}
        settings = dict(r1_ref=50, r2_ref=10, b_ref=4, r2_lambda=2, r3_beta=.5)
        expected = {"r1": -.5, "r2": -2, "r3": -1.5, "r4": -.3}
        for mode, reward in expected.items():
            env = GamaTrafficEnv(reward_mode=mode, state_mode="spatial", **settings)
            self.assertAlmostEqual(env.transition_reward(old, new), reward)
        state = {"spatial": observation()["grid"], "global_phase": 4,
                 "green_cycles": 300, "queue_ns": 20, "queue_ew": 10,
                 "approach_ns": 5, "approach_ew": 7,
                 "pressure": np.array([2, -1, 4, 0], dtype=np.float32),
                 "action_mask": np.array([1, 1, 0, 1], dtype=np.bool_)}
        features = featurize(state)
        self.assertEqual(features["grid"].shape, (4, 6, 20, 16))
        self.assertEqual(features["aux"][2], 1)
        self.assertEqual(features["aux"][15], 0)
        self.assertAlmostEqual(float(features["aux"][9]), 2 / 50)
        np.testing.assert_array_equal(features["pressure"], state["pressure"])

    def test_previous_thirteen_aux_checkpoint_still_evaluates(self):
        old = RainbowAgent(aux_size=13, buffer_size=1, batch_size=1, warmup=1,
                           device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "previous.pt"
            old.save(path)
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["architecture"] = PREVIOUS_ARCHITECTURE
            torch.save(checkpoint, path)
            loaded = RainbowAgent(aux_size=13, buffer_size=1, batch_size=1,
                                  warmup=1, device=torch.device("cpu"))
            loaded.load(path)
            action = loaded.act(observation(), greedy=True, action_mask=[1, 0, 0, 0])
        self.assertEqual(action, 0)


class SpatialRolloutEdgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_waits_until_cycle_stops(self):
        env = GamaTrafficEnv()
        reads = iter((10, 11, 11, 11))
        pauses = []

        async def pause():
            pauses.append(True)

        async def read_cycle(expression):
            self.assertEqual(expression, "cycle")
            return next(reads)

        env.pause = pause
        env._exec_expression = read_cycle
        await env.pause_until_stable()
        self.assertEqual(len(pauses), 2)

    async def test_horizon_mid_action_is_not_stored_as_terminal(self):
        class FakeEnv:
            def __init__(self):
                self.cycle = 0

            def snapshot(self):
                result = {"cycle": self.cycle, "dqn_waiting_action": self.cycle in (100, 320),
                          "next_decision_in_cycles": 100, "spatial": observation()["grid"],
                          "pressure": np.zeros(4, dtype=np.float32),
                          "global_phase": 0, "green_cycles": 100,
                          "queue_ns": 0, "queue_ew": 0, "approach_ns": 0,
                          "approach_ew": 0,
                          "action_mask": np.ones(4, dtype=np.bool_)}
                return result

            async def reset(self, **kwargs):
                return self.snapshot()

            async def step_cycles(self, count):
                self.cycle += count

            async def read_state(self):
                return self.snapshot()

            async def step(self, action, horizon, current_state):
                self.cycle = 320 if self.cycle == 100 else horizon
                valid = self.cycle == 320
                return self.snapshot(), -1.0, not valid, {
                    "bootstrap_valid": valid, "elapsed_seconds": 22.0 if valid else 8.0}

        class FakeActor:
            def act(self, observation, **kwargs):
                return 0

            def decay_epsilon(self):
                pass

        args = SimpleNamespace(episode_horizon=400, algorithm="rainbow", warmup=0)
        result = await collect_rollout(FakeEnv(), FakeActor(), args, 1, 2, False)
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(len(result["transitions"]), 1)
        self.assertFalse(result["transitions"][0][-1])
        expired = await collect_rollout(FakeEnv(), FakeActor(), args, 1, 2, False,
                                        deadline=0.0)
        self.assertEqual(expired["decisions"], 0)
        self.assertEqual(expired["transitions"], [])

    async def test_worker_rollout_continues_same_episode_without_reset(self):
        class FakeEnv:
            def __init__(self):
                self.cycle = 0
                self.resets = 0

            def snapshot(self):
                return {"cycle": self.cycle, "dqn_waiting_action": True,
                        "next_decision_in_cycles": 100,
                        "spatial": observation()["grid"],
                        "pressure": np.zeros(4, dtype=np.float32),
                        "global_phase": 0, "green_cycles": 100,
                        "queue_ns": 0, "queue_ew": 0, "approach_ns": 0,
                        "approach_ew": 0,
                        "action_mask": np.ones(4, dtype=np.bool_)}

            async def reset(self, **kwargs):
                self.resets += 1
                self.cycle = 0
                return self.snapshot()

            async def step_cycles(self, count):
                self.cycle += count

            async def read_state(self):
                return self.snapshot()

            async def step(self, action, horizon, current_state):
                self.cycle += 100
                return self.snapshot(), -1.0, False, {
                    "bootstrap_valid": True, "elapsed_seconds": 10.0}

        class FakeActor:
            def act(self, observation, **kwargs):
                return 0

            def decay_epsilon(self):
                pass

        env = FakeEnv()
        args = SimpleNamespace(episode_horizon=800, algorithm="rainbow", warmup=0)
        first = await collect_rollout(env, FakeActor(), args, 1001, 2, False)
        second = await collect_rollout(env, FakeActor(), args, 1001, 2, False,
                                       initial_state=first["state"])
        self.assertEqual(env.resets, 1)
        self.assertEqual(first["simulated_cycles"], 300)
        self.assertEqual(second["simulated_cycles"], 200)
        self.assertEqual(second["state"]["cycle"], 500)
        self.assertEqual(len(first["transitions"]), 2)
        self.assertEqual(len(second["transitions"]), 2)


if __name__ == "__main__":
    unittest.main()
