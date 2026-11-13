"""Learning-contract tests for the four-phase tabular and flat DQN baselines."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train.logic.tabular_agent import TabularQAgent, state_key
from train.logic.three_dqn_agent import FlatDQNAgent, FlatQNetwork


class FourPhaseAgentTest(unittest.TestCase):
    def setUp(self):
        self.state = np.zeros((3, 8, 20), dtype=np.float32)
        self.next_state = self.state.copy()
        self.next_state[0, 2, :4] = 1
        self.next_state[1, 2, :4] = 0.05
        self.next_state[2, 2] = 1

    def test_tabular_uses_mask_and_elapsed_discount(self):
        agent = TabularQAgent(lr=1.0, gamma=0.5, reward_mode="queue_time", seed=7)
        mask = [False, True, False, False]
        self.assertEqual(agent.act(self.state, action_mask=mask, force_random=True), 1)
        key2 = state_key(self.next_state)
        agent.table[key2] = np.asarray([20, 4, 30, 40], dtype=np.float32)
        agent.store(self.state, 1, 1.0, self.next_state, False,
                    seconds=10.0, next_mask=mask)
        self.assertAlmostEqual(agent.update(), 2.0)
        self.assertAlmostEqual(agent.table[state_key(self.state)][1], 2.0)
        self.assertEqual(agent.act(self.state, greedy=True, action_mask=mask), 1)

    def test_flat_dqn_updates_and_roundtrips_checkpoint(self):
        agent = FlatDQNAgent(buffer_size=4, warmup=1, batch_size=1,
                             seed=5, device=torch.device("cpu"))
        self.assertEqual(tuple(FlatQNetwork()(torch.from_numpy(self.state[None])).shape), (1, 4))
        agent.store(self.state, 2, 0.5, self.next_state, False,
                    next_mask=[False, False, True, False])
        self.assertTrue(np.isfinite(agent.update()))
        self.assertEqual(agent.update_count, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flat.pt"
            agent.save(path)
            loaded = FlatDQNAgent(buffer_size=1, warmup=1,
                                  device=torch.device("cpu"))
            loaded.load(path)
            self.assertEqual(loaded.act(self.next_state, greedy=True,
                                        action_mask=[False, True, False, False]), 1)
            for before, after in zip(agent.online_net.parameters(),
                                     loaded.online_net.parameters()):
                torch.testing.assert_close(before, after)

    def test_tabular_checkpoint_preserves_policy(self):
        agent = TabularQAgent(seed=3)
        agent.table[state_key(self.state)] = np.asarray([0, 2, 1, 0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.pt"
            agent.save(path)
            loaded = TabularQAgent(seed=99)
            loaded.load(path)
            self.assertEqual(loaded.act(self.state, greedy=True), 1)

    def test_training_resume_restores_replay_optimizer_and_rng(self):
        agent = FlatDQNAgent(buffer_size=4, warmup=1, batch_size=1,
                             seed=13, device=torch.device("cpu"))
        agent.store(self.state, 2, 1.0, self.next_state, False,
                    next_mask=[True, False, True, False])
        agent.update()
        snapshot = agent.training_state()
        loaded = FlatDQNAgent(buffer_size=4, warmup=1, batch_size=1,
                              seed=99, device=torch.device("cpu"))
        loaded.restore_training(snapshot)
        self.assertEqual(loaded.buffer.size, agent.buffer.size)
        self.assertEqual(loaded.update_count, agent.update_count)
        self.assertEqual(loaded.act(self.state, action_mask=[False, True, False, False],
                                    force_random=True), 1)
        self.assertAlmostEqual(loaded.update(), agent.update(), places=6)
        table = TabularQAgent(seed=3)
        table.store(self.state, 1, 2.0, self.next_state, False,
                    next_mask=[False, True, False, False])
        table.update()
        restored = TabularQAgent(seed=8)
        restored.restore_training(table.training_state())
        self.assertEqual(restored.decisions, table.decisions)
        self.assertEqual(restored.update_count, table.update_count)
        np.testing.assert_array_equal(restored.table[state_key(self.state)],
                                      table.table[state_key(self.state)])


if __name__ == "__main__":
    unittest.main()
