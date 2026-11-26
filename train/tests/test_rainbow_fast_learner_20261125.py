"""The fast Rainbow learner (--fast-learner) must learn exactly like the eager one.

* uint8 replay storage: bit-identical samples and snapshot bytes.
* merged online forward / CUDA graph: same update within float tolerance,
  with the same noise stream and the same PER order.
* checkpoints move freely between the two learners.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from train.logic.rainbow_agent import GRID_SHAPE, AUX_SIZE, PrioritizedReplay, RainbowAgent
from train.logic.train import DemandManifest, check_resume_config, parse_args

MASK = np.array([True, False, True, False])


def observation(rng):
    grid = np.zeros(GRID_SHAPE, dtype=np.float32)
    occupied = rng.random(GRID_SHAPE) < .08
    grid[occupied] = rng.random(int(occupied.sum()))
    grid[:, 4] = np.where(occupied[:, 4], grid[:, 4] * 2 - 1, 0)
    return {"grid": grid, "aux": rng.random(AUX_SIZE).astype(np.float32)}


def filled_replays(n=50, capacity=32):
    zlib_replay = PrioritizedReplay(capacity, alpha=.5, seed=11)
    raw_replay = PrioritizedReplay(capacity, alpha=.5, seed=11)
    raw_replay.use_raw_observations()
    rng = np.random.default_rng(5)
    for i in range(n):  # wraps around the capacity
        transition = (observation(rng), int(rng.choice([0, 2])), -float(rng.random() * 30),
                      observation(rng), bool(i % 17 == 16), float(rng.uniform(.5, 1)), MASK)
        zlib_replay.push(*transition)
        raw_replay.push(*transition)
    return zlib_replay, raw_replay


def assert_same_sample(test, a, b):
    for left, right in zip(a, b):
        test.assertEqual(np.asarray(left).dtype, np.asarray(right).dtype)
        test.assertTrue(np.array_equal(left, right))


def make_agent(fast, device="cpu", seed=3):
    agent = RainbowAgent(buffer_size=256, warmup=0, batch_size=8, atoms=51, v_min=-500.0,
                         n_step=3, target_update=5, device=torch.device(device), seed=seed)
    if fast:
        agent.enable_fast_learner()
    rng = np.random.default_rng(7)
    for i in range(80):
        agent.store(observation(rng), int(rng.choice([0, 2])), -float(rng.random() * 20),
                    observation(rng), i % 30 == 29, seconds=10.0, next_mask=MASK)
    agent.finish_episode()  # flush the n-step tail so training_state() is allowed
    return agent


def run_updates(fast, steps, device="cpu"):
    agent = make_agent(fast, device)  # the constructor reseeds torch: same noise stream
    losses = [agent.update() for _ in range(steps)]
    return agent, losses


class FastReplayTests(unittest.TestCase):
    def test_raw_storage_samples_are_bit_identical(self):
        zlib_replay, raw_replay = filled_replays()
        zlib_replay.update_priorities(np.arange(8), np.linspace(.1, 5, 8))
        raw_replay.update_priorities(np.arange(8), np.linspace(.1, 5, 8))
        for beta in (.4, .7, 1.0):
            assert_same_sample(self, zlib_replay.sample(16, beta), raw_replay.sample(16, beta))

    def test_snapshot_bytes_unchanged_and_restorable_both_ways(self):
        zlib_replay, raw_replay = filled_replays()
        old, new = zlib_replay.snapshot(), raw_replay.snapshot()
        self.assertEqual(old.keys(), new.keys())
        self.assertEqual(old["states"], new["states"])
        self.assertEqual(old["next_states"], new["next_states"])
        for key in ("aux", "next_aux", "actions", "rewards", "discounts", "dones",
                    "next_masks", "tree_sum", "tree_min"):
            self.assertTrue(np.array_equal(old[key], new[key]), key)

        into_raw = PrioritizedReplay(32, alpha=.5, seed=99)
        into_raw.use_raw_observations()
        into_raw.restore(old)
        into_zlib = PrioritizedReplay(32, alpha=.5, seed=99)
        into_zlib.restore(new)
        assert_same_sample(self, into_raw.sample(16, .5), into_zlib.sample(16, .5))

    def test_storage_switch_refuses_a_filled_replay(self):
        zlib_replay, _ = filled_replays(n=3)
        with self.assertRaises(ValueError):
            zlib_replay.use_raw_observations()


class FastUpdateTests(unittest.TestCase):
    def setUp(self):
        # Compare the update rule in true FP32. With TF32 convolutions (PyTorch's
        # default) cuDNN may pick different algorithms for the merged batch of 64
        # and the separate batches of 32 (it does on an RTX 5090), and Adam blows
        # those ~1e-3 rounding differences up on near-zero gradients.
        self.tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = self.tf32

    def check_matches_reference(self, device, steps, rtol):
        reference, ref_losses = run_updates(False, steps, device)
        fast, fast_losses = run_updates(True, steps, device)
        np.testing.assert_allclose(fast_losses, ref_losses, rtol=rtol, atol=1e-6)
        for (name, ref), (_, new) in zip(reference.online_net.state_dict().items(),
                                         fast.online_net.state_dict().items()):
            np.testing.assert_allclose(new.cpu().numpy(), ref.cpu().numpy(),
                                       rtol=rtol, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(fast.buffer.tree.sum, reference.buffer.tree.sum, rtol=rtol)
        self.assertEqual(fast.update_count, reference.update_count)
        return fast

    def test_merged_forward_matches_eager_update_on_cpu(self):
        self.check_matches_reference("cpu", steps=20, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_graph_matches_eager_update(self):
        fast = self.check_matches_reference("cuda", steps=RainbowAgent.GRAPH_WARMUP + 17, rtol=1e-4)
        self.assertIsNotNone(fast._graph)

    def test_checkpoints_move_between_learners(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for source_fast in (False, True):
                source, _ = run_updates(source_fast, 6, device)
                state = source.training_state()
                for group in state["optimizer"]["param_groups"]:
                    self.assertFalse(group.get("capturable", False))
                target = make_agent(not source_fast, device, seed=9)
                target.restore_training(state)
                self.assertEqual(target.update_count, source.update_count)
                assert_same_sample(self, target.buffer.sample(8, .5), source.buffer.sample(8, .5))
                for _ in range(RainbowAgent.GRAPH_WARMUP + 3):  # recaptures after a restore
                    self.assertTrue(np.isfinite(target.update()))


class FastLearnerFlagTests(unittest.TestCase):
    def test_fast_learner_is_rainbow_only(self):
        with self.assertRaises(SystemExit):
            parse_args(["--algorithm", "hybrid", "--reward-mode", "r1", "--fast-learner"])
        args = parse_args(["--algorithm", "rainbow", "--reward-mode", "r1", "--fast-learner"])
        self.assertTrue(args.fast_learner)

    def test_allow_code_change_only_relaxes_source_hashes(self):
        saved = {"code_sha256": {"train.py": "a"}, "lr": 1e-4, "workers": 2}
        current = {"code_sha256": {"train.py": "b"}, "lr": 1e-4, "workers": 40}
        with self.assertRaises(ValueError):
            check_resume_config(saved, current)
        self.assertEqual(check_resume_config(saved, current, allow_code_change=True),
                         ["code_sha256", "workers"])
        with self.assertRaises(ValueError):
            check_resume_config(saved, {**current, "lr": 2e-4}, allow_code_change=True)


class DemandManifestTests(unittest.TestCase):
    """demand_manifest.json is written at checkpoints, byte-identical to the old per-episode write."""

    def old_write(self, entries, path):
        with open(path, "w") as stream:
            json.dump(entries, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def test_record_keeps_memory_only_and_flush_writes_the_old_bytes(self):
        with tempfile.TemporaryDirectory() as outdir:
            path = os.path.join(outdir, "demand_manifest.json")
            manifest = DemandManifest(path)
            flows = {"north": 412.12345, "east": 300.0}
            manifest.record(7, os.path.join(outdir, "demand", "seed_7.csv"), "ab",
                            outdir, (0.912345678, 0.1234567, flows))
            manifest.record(3, os.path.join(outdir, "demand", "seed_3.csv"), "cd", outdir)
            self.assertFalse(os.path.exists(path))
            self.assertTrue(manifest.flush())
            expected = {"7": {"path": os.path.join("demand", "seed_7.csv"), "sha256": "ab",
                              "scale": 0.912346, "axis_imbalance": 0.123457,
                              "flow_veh_h": {"north": 412.123, "east": 300.0}},
                        "3": {"path": os.path.join("demand", "seed_3.csv"), "sha256": "cd"}}
            reference = os.path.join(outdir, "reference.json")
            self.old_write(expected, reference)
            with open(path, "rb") as new, open(reference, "rb") as old:
                self.assertEqual(new.read(), old.read())
            self.assertFalse(manifest.flush())  # nothing new: no rewrite
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_resume_loads_and_extends_the_saved_manifest(self):
        with tempfile.TemporaryDirectory() as outdir:
            path = os.path.join(outdir, "demand_manifest.json")
            self.old_write({"1": {"path": "demand/seed_1.csv", "sha256": "x"}}, path)
            manifest = DemandManifest.load(path)
            manifest.record(2, os.path.join(outdir, "demand", "seed_2.csv"), "y", outdir)
            manifest.flush()
            with open(path) as stream:
                self.assertEqual(sorted(json.load(stream)), ["1", "2"])


if __name__ == "__main__":
    unittest.main()
