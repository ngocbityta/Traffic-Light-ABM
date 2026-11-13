"""Checks for the DTSE representation and 3DQN learning path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train.logic.three_dqn_agent import (
    DuelingQNetwork, ThreeDQNAgent, pack_state, unpack_states,
)


class ThreeDQNAgentTest(unittest.TestCase):
    def test_exploration_reaches_reference_minimum_at_decay_step(self):
        agent = ThreeDQNAgent(buffer_size=1, warmup=1, eps_start=1.0,
                              eps_min=0.01, eps_decay_steps=4,
                              seed=7, device=torch.device("cpu"))
        for _ in range(2):
            agent.decay_epsilon()
        self.assertAlmostEqual(agent.eps, np.exp(-0.5))
        for _ in range(2):
            agent.decay_epsilon()
        self.assertAlmostEqual(agent.eps, np.exp(-1.0))
        for _ in range(100):
            agent.decay_epsilon()
        agent.decay_epsilon()
        self.assertAlmostEqual(agent.eps, 0.01)

    def test_warmup_can_force_uniform_random_legal_actions(self):
        agent = ThreeDQNAgent(buffer_size=1, warmup=1, eps_start=0.0,
                              eps_min=0.0, seed=7, device=torch.device("cpu"))
        state = np.zeros((3, 8, 20), dtype=np.float32)
        mask = [False, True, False, False]
        for _ in range(10):
            self.assertEqual(agent.act(state, action_mask=mask,
                                       force_random=True), 1)

    def test_dtse_pack_roundtrip_and_network_output(self):
        state = np.zeros((3, 8, 20), dtype=np.float32)
        state[0, 2, 4] = 1
        state[1, 2, 4] = 0.47
        state[2, 2] = 1
        state[2, 6] = 1
        restored = unpack_states(pack_state(state)[None])[0]
        np.testing.assert_array_equal(restored[[0, 2]], state[[0, 2]])
        np.testing.assert_allclose(restored[1], state[1], atol=1 / 255)
        self.assertEqual(tuple(DuelingQNetwork()(torch.from_numpy(state[None])).shape),
                         (1, 4))

    def test_replay_gradient_soft_target_and_checkpoint(self):
        agent = ThreeDQNAgent(buffer_size=4, warmup=2, batch_size=2,
                              tau=0.25, seed=71, device=torch.device("cpu"))
        state = np.zeros((3, 8, 20), dtype=np.float32)
        next_state = state.copy()
        next_state[0, 3, 4] = 1
        next_state[1, 3, 4] = 0.3
        next_state[2, 3] = 1
        agent.store(state, 1, 0.5, next_state, False)
        self.assertFalse(agent.can_update())
        agent.store(next_state, 2, 0.2, state, True)
        self.assertTrue(agent.can_update())
        before = [p.detach().clone() for p in agent.target_net.parameters()]
        loss = agent.update()
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.update_count, 1)
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before, agent.online_net.parameters())))
        for old, online, target in zip(before, agent.online_net.parameters(),
                                       agent.target_net.parameters()):
            torch.testing.assert_close(target, old * 0.75 + online * 0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            agent.save(path)
            restored = ThreeDQNAgent(buffer_size=1, warmup=1,
                                      device=torch.device("cpu"))
            restored.load(path)
            for original, loaded in zip(agent.online_net.parameters(),
                                        restored.online_net.parameters()):
                torch.testing.assert_close(original, loaded)

    def test_double_target_uses_online_argmax_and_target_value(self):
        agent = ThreeDQNAgent(buffer_size=1, warmup=1, batch_size=1,
                              gamma=1.0, seed=3, device=torch.device("cpu"))
        with torch.no_grad():
            for network in (agent.online_net, agent.target_net):
                for parameter in network.parameters():
                    parameter.zero_()
            agent.online_net.advantage.bias[0] = 2.0
            agent.target_net.advantage.bias[1] = 3.0
        state = np.zeros((3, 8, 20), dtype=np.float32)
        agent.store(state, 0, 0.0, state, False)
        # Online Q(0)=1.5, online argmax=0. Target Q(0)=-0.75,
        # whereas target argmax=1 has Q(1)=2.25. Huber(1.5,-0.75)=1.75.
        self.assertAlmostEqual(agent.update(), 1.75, places=5)

    def test_mask_and_elapsed_time_are_used_in_double_target(self):
        agent = ThreeDQNAgent(buffer_size=1, warmup=1, batch_size=1,
                              gamma=.5, reward_mode="queue_time", device=torch.device("cpu"))
        with torch.no_grad():
            for network in (agent.online_net, agent.target_net):
                for parameter in network.parameters():
                    parameter.zero_()
            agent.online_net.advantage.bias[0] = 2.0
            agent.target_net.advantage.bias[1] = 3.0
        state = np.zeros((3, 8, 20), dtype=np.float32)
        mask = [False, True, False, False]
        self.assertEqual(agent.act(state, greedy=True, action_mask=mask), 1)
        for _ in range(10):
            self.assertEqual(agent.act(state, action_mask=mask), 1)
        agent.store(state, 0, 0., state, False, seconds=10., next_mask=mask)
        # Q(s,0)=1.5; legal next action 1 has target value 2.25.
        # Time discount=.5**(10/5)=.25; Huber(1.5,.5625)=.439453125.
        self.assertAlmostEqual(agent.update(), .439453125, places=6)


if __name__ == "__main__":
    unittest.main()
