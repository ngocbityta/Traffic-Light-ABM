"""Hybrid 3DQN + Max Pressure action and replay behavior."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train.logic.hybrid_agent import SpatialHybridAgent


def state(pressure):
    grid = np.zeros((4, 6, 20, 16), dtype=np.float32)
    grid[:, 5] = 1.0
    grid[1, 0, 2, 3] = .5
    return {"grid": grid, "aux": np.zeros(17, dtype=np.float32),
            "pressure": np.asarray(pressure, dtype=np.float32)}


class HybridTests(unittest.TestCase):
    def test_pressure_can_change_action_and_mask_still_wins(self):
        agent = SpatialHybridAgent(buffer_size=4, batch_size=1, warmup=0,
                                   pressure_lambda=2, device=torch.device("cpu"))
        q = torch.tensor([[.8, .6, 0., 0.]])
        p = torch.tensor([[.2, .9, 0., 0.]])
        legal = torch.tensor([[True, True, False, False]])
        self.assertEqual(int(agent._scores(q, p, legal).argmax()), 1)
        self.assertEqual(agent.act(state([1, 2, 3, 4]), greedy=True,
                                   action_mask=[0, 0, 1, 0]), 2)

    def test_update_and_checkpoint_preserve_pressure_and_reward(self):
        agent = SpatialHybridAgent(buffer_size=8, batch_size=2, warmup=2,
                                   reward_mode="r4", pressure_lambda=.75,
                                   reward_settings={"r1_ref": 80.0},
                                   device=torch.device("cpu"), seed=9)
        first = state([1, -2, 3, 0])
        second = state([0, 4, -1, 2])
        for i in range(3):
            agent.store(first, i, -i - 1, second, i == 2,
                        seconds=10, next_mask=[1, 1, 0, 1])
        self.assertEqual(len(agent.buffer), 3)
        np.testing.assert_array_equal(agent.buffer.next_pressures[0], second["pressure"])
        self.assertAlmostEqual(float(agent.buffer.discounts[0]), agent.gamma ** 2)
        before = next(agent.online_net.parameters()).detach().clone()
        self.assertTrue(math.isfinite(agent.update()))
        self.assertFalse(torch.equal(before, next(agent.online_net.parameters()).detach()))
        snapshot = agent.training_state()
        restored = SpatialHybridAgent(buffer_size=8, batch_size=2, warmup=2,
                                      reward_mode="r4", device=torch.device("cpu"))
        restored.restore_training(snapshot)
        self.assertEqual(len(restored.buffer), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.pt"
            agent.save(path)
            restored.load(path)
        self.assertEqual(restored.reward_mode, "r4")
        self.assertEqual(restored.pressure_lambda, .75)
        self.assertEqual(restored.reward_settings["r1_ref"], 80.0)


if __name__ == "__main__":
    unittest.main()
