"""Learner thread, weight-only actor payloads and vectorised PER (21/11/2026)."""
import pickle
import random
import unittest

import numpy as np
import torch

from train.logic.rainbow_agent import PrioritizedReplay, RainbowAgent
from train.logic import train as train_module
from train.logic.train import (LearnerThread, actor_seed, learn_emitted, load_actor_payload,
                               make_actor, parse_args, weights_payload)

ARGS = ["--scenario", "lvl", "--backend", "python", "--algorithm", "rainbow",
        "--reward-mode", "r1", "--workers", "2", "--warmup", "8", "--batch-size", "4"]


def observation(rng):
    grid = (rng.random((4, 6, 20, 16)) < 0.2).astype(np.float32)
    grid[:, 4] = 0.0
    return {"grid": grid, "aux": rng.random(17).astype(np.float32),
            "pressure": np.zeros(4, np.float32)}


def rollouts(count=6, length=5, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.array([True, False, True, False])
    out = []
    for _ in range(count):
        out.append([(observation(rng), int(rng.integers(2)) * 2, float(-rng.random() * 30),
                     observation(rng), 10.0, mask, False) for _ in range(length)])
    return out


def agent_for(args):
    torch.manual_seed(7)
    return RainbowAgent(lr=args.lr, gamma=args.gamma, buffer_size=200, warmup=args.warmup,
                        batch_size=args.batch_size, seed=7, device=torch.device("cpu"),
                        **train_module.spatial_kwargs(args))


class LearnerThreadTests(unittest.TestCase):
    def test_thread_learns_like_the_dispatch_loop(self):
        args = parse_args(ARGS)
        data = rollouts()
        inline = agent_for(args)
        train_module._UPDATE_CREDIT[0] = 0.0
        for transitions in data:
            for o, a, r, o2, s, m, d in transitions:
                learn_emitted(inline, args, inline.store(o, a, r, o2, d, seconds=s, next_mask=m))
            learn_emitted(inline, args, inline.finish_episode())

        threaded = agent_for(args)
        train_module._UPDATE_CREDIT[0] = 0.0
        learner = LearnerThread(threaded, args)
        learner.start()
        for transitions in data:
            learner.submit(transitions)
        learner.stop()
        learner.join(timeout=60)
        learner.check()
        self.assertEqual(learner.pending, 0)
        self.assertEqual(threaded.update_count, inline.update_count)
        self.assertGreater(threaded.update_count, 0)
        for (name, a), (_, b) in zip(inline.online_net.state_dict().items(),
                                      threaded.online_net.state_dict().items()):
            self.assertTrue(torch.equal(a, b), name)
        self.assertTrue(np.array_equal(inline.buffer.tree.sum, threaded.buffer.tree.sum))

    def test_errors_reach_the_dispatcher(self):
        args = parse_args(ARGS)
        learner = LearnerThread(agent_for(args), args)
        learner.start()
        learner.submit([("not an observation",)])
        learner.join(timeout=30)
        with self.assertRaises(RuntimeError):
            learner.check()


class WeightPayloadTests(unittest.TestCase):
    def test_weight_payload_acts_like_a_pickled_actor(self):
        args = parse_args(ARGS)
        learner = agent_for(args)
        seed = actor_seed(1003, 160, 1)
        reference = pickle.loads(pickle.dumps(make_actor(RainbowAgent, learner, args, seed)))
        torch.manual_seed(seed)  # what the payload path does in the worker
        train_module._PROCESS_ACTOR = None
        payload = weights_payload(learner, seed)
        cached = load_actor_payload(payload, args)
        # A second load reuses the process actor and resets its streams.
        cached = load_actor_payload(payload, args)
        self.assertLess(len(payload[1]), 2_500_000)
        for (name, a), (_, b) in zip(reference.online_net.state_dict().items(),
                                      cached.online_net.state_dict().items()):
            if "epsilon" not in name:  # NoisyNet noise is redrawn before every action
                self.assertTrue(torch.equal(a, b), name)
        rng = np.random.default_rng(3)
        mask = np.array([True, False, True, False])
        states = [observation(rng) for _ in range(5)]
        torch.manual_seed(seed)
        expected = [reference.act(s, action_mask=mask) for s in states]
        expected += [reference.act(s, action_mask=mask, force_random=True) for s in states]
        cached = load_actor_payload(payload, args)
        got = [cached.act(s, action_mask=mask) for s in states]
        got += [cached.act(s, action_mask=mask, force_random=True) for s in states]
        self.assertEqual(got, expected)


class VectorisedReplayTests(unittest.TestCase):
    def test_sampling_and_priorities_match_the_sequential_tree(self):
        rng = np.random.default_rng(5)
        a, b = PrioritizedReplay(1000, .5, seed=3), PrioritizedReplay(1000, .5, seed=3)
        priorities = rng.random(700) * 3
        for i, p in enumerate(priorities):
            a.tree.set(i, p ** .5)
            b.tree.set(i, p ** .5)
        a.size = b.size = 700
        for _ in range(100):
            segment = a.tree.sum[1] / 32
            sequential = [a.tree.find((i + a.rng.random()) * segment) for i in range(32)]
            vector = b.tree.find_many((np.arange(32) + b.rng.random(32)) * (b.tree.sum[1] / 32))
            self.assertEqual(list(vector), sequential)
            losses = (rng.standard_normal(32) * 5).astype(np.float32)
            index = np.array(sequential)
            index[:3] = index[0]  # repeated index: the last value wins
            for i, loss in zip(index, losses):
                priority = float(abs(loss)) + 1e-6
                a.max_priority = max(a.max_priority, priority)
                a.tree.set(int(i), priority ** a.alpha)
            b.update_priorities(index, losses)
            self.assertTrue(np.array_equal(a.tree.sum, b.tree.sum))
            self.assertTrue(np.array_equal(a.tree.minimum, b.tree.minimum))
            self.assertEqual(a.max_priority, b.max_priority)



class ResumeAcrossMachinesTests(unittest.TestCase):
    def test_run_settings_may_change_but_learning_settings_may_not(self):
        from train.logic.train import check_resume_config
        saved = {"workers": 6, "device": None, "lr": 1e-4, "code_sha256": {"a": "1"}}
        moved = dict(saved, workers=46, device="cuda", learner="thread")
        self.assertEqual(check_resume_config(saved, moved), ["device", "learner", "workers"])
        with self.assertRaises(ValueError):
            check_resume_config(saved, dict(saved, lr=3e-4))
        with self.assertRaises(ValueError):
            check_resume_config(saved, dict(saved, code_sha256={"a": "2"}))


if __name__ == "__main__":
    unittest.main()
