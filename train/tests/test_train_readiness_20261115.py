"""Training-pipeline regressions for the 15/11/2026 fixes."""
import unittest

import numpy as np
import torch

from train.logic.hybrid_agent import SpatialHybridAgent, normalized_scores
from train.logic.lvl_env import MIN_GREEN_CYCLES, _observation, action_mask
from train.logic.gama_env import featurize
from train.logic.train import actor_seed, make_actor, parse_args


class HybridScoreTests(unittest.TestCase):
    def setUp(self):
        self.legal = torch.tensor([[True, False, True, False]])

    def test_two_action_z_score_ignored_the_size_of_the_gap(self):
        tiny = normalized_scores(torch.tensor([[1.00, 0., 1.01, 0.]]), self.legal)
        huge = normalized_scores(torch.tensor([[1.0, 0., 50.0, 0.]]), self.legal)
        self.assertTrue(torch.allclose(tiny[:, [0, 2]], huge[:, [0, 2]]))

    def test_large_pressure_gap_outvotes_a_small_q_gap(self):
        agent = SpatialHybridAgent(buffer_size=4, batch_size=1, warmup=0,
                                   pressure_lambda=.5, device=torch.device("cpu"))
        agent.q_spread, agent.pressure_spread = 1.0, 5.0
        q = torch.tensor([[1.05, 0., 1.00, 0.]])         # Q barely prefers action 0
        p = torch.tensor([[2.0, 0., 30.0, 0.]])          # pressure strongly prefers 2
        self.assertEqual(int(agent._scores(q, p, self.legal).argmax()), 2)
        q = torch.tensor([[4.0, 0., 1.0, 0.]])           # a clear Q preference holds
        self.assertEqual(int(agent._scores(q, p, self.legal).argmax()), 0)
        self.assertEqual(int(agent._scores(q, p, torch.tensor(
            [[True, False, False, False]])).argmax()), 0)

    def test_spreads_survive_training_state_round_trip(self):
        agent = SpatialHybridAgent(buffer_size=4, batch_size=1, warmup=0,
                                   device=torch.device("cpu"))
        agent.q_spread, agent.pressure_spread = 0.7, 3.0
        other = SpatialHybridAgent(buffer_size=4, batch_size=1, warmup=0,
                                   device=torch.device("cpu"))
        other.restore_training(agent.training_state())
        self.assertEqual((other.q_spread, other.pressure_spread), (0.7, 3.0))


class ActorTests(unittest.TestCase):
    def test_each_rollout_of_an_episode_gets_its_own_exploration_stream(self):
        seeds = {actor_seed(1001, decisions, worker)
                 for decisions in (0, 16, 32) for worker in (0, 1)}
        self.assertEqual(len(seeds), 6)

    def test_building_an_actor_does_not_rewind_the_learner_generators(self):
        args = parse_args(["--algorithm", "hybrid", "--total-steps", "100"])
        learner = SpatialHybridAgent(buffer_size=8, batch_size=1, warmup=0,
                                     device=torch.device("cpu"), seed=3)
        torch.manual_seed(123)
        np.random.seed(123)
        expected = (torch.rand(3), np.random.rand(3))
        torch.manual_seed(123)
        np.random.seed(123)
        make_actor(SpatialHybridAgent, learner, args, actor_seed(1001, 0))
        self.assertTrue(torch.equal(torch.rand(3), expected[0]))
        np.testing.assert_array_equal(np.random.rand(3), expected[1])

    def test_epsilon_decay_defaults_to_the_run_length(self):
        args = parse_args(["--algorithm", "hybrid", "--total-steps", "1000000"])
        self.assertEqual(args.eps_decay_steps, 200000.0)
        self.assertEqual(parse_args(["--algorithm", "hybrid", "--total-steps", "100",
                                     "--eps-decay-steps", "7"]).eps_decay_steps, 7.0)


class ObservationTests(unittest.TestCase):
    def observe(self, phase, green, **kw):
        return _observation(100, phase, True, 1, 0, 0, 0.0, 0.0, 0.0, [], "spatial",
                            green_cycles=green, **kw)

    def test_minimum_green_mask(self):
        self.assertEqual(action_mask("LVL", MIN_GREEN_CYCLES - 1).tolist(),
                         [True, False, False, False])
        self.assertEqual(action_mask("HDT", MIN_GREEN_CYCLES).tolist(),
                         [True, False, True, False])
        self.assertEqual(self.observe("HDT", 20)["action_mask"].tolist(),
                         [False, False, True, False])

    def test_transition_is_not_featurized_as_lvl_green(self):
        leaving_hdt = featurize(self.observe("none", 0, transition_of="HDT"))["aux"]
        lvl_green = featurize(self.observe("LVL", 60))["aux"]
        self.assertEqual(leaving_hdt[:4].tolist(), [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(lvl_green[:4].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.observe("none", 0, transition_of="HDT")["global_phase"], 5)


if __name__ == "__main__":
    unittest.main()
