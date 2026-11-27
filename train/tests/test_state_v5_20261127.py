"""State v5 (50 m camera grid 4x7x10x12 + aux 30) with the +5 s extension
control, through the training adapter and the Rainbow agent (27/11/2026)."""

import asyncio
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from python_simulation.demand import axis_flows, generate_demand
from train.logic.gama_env import featurize
from train.logic.lvl_env import ARMS, DT, LVLPythonEnv, V5_AUX_SIZE, V5_GRID_SHAPE
from train.logic import rainbow_agent as ra
from train.logic.rainbow_agent import RainbowAgent

SIM = {"yellow": 4.0, "all_red": 0.0, "min_green": 15.0, "signal_control": "extend",
       "max_green": 75.0}


def make_env(**kwargs):
    return LVLPythonEnv(state_mode="spatial", reward_mode="r4", state_version="v5",
                        sim_config=dict(SIM), **kwargs)


class StateV5EnvTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.demand = Path(cls.tmp.name) / "d.csv"
        generate_demand(cls.demand, 11, 1500, heavy_ratio=.03,
                        flow_by_arm=axis_flows(1.1, .1))
        cls.parameters = [{"name": "demand_file_path", "type": "string",
                           "value": str(cls.demand)}]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    async def episode(self, decisions=40, seed=11, policy=None):
        env = make_env()
        state = await env.reset(seed=seed, parameters=self.parameters)
        await env.step_cycles(state["next_decision_in_cycles"])
        state = await env.read_state()
        rng = random.Random(seed)
        out = []
        for _ in range(decisions):
            self.assertTrue(state["dqn_waiting_action"])
            obs = featurize(state)
            action = policy(env.sim) if policy else rng.choice((0, 2))
            out.append((obs, env.sim.phase, env.sim.green_ticks, env.sim.planned_green_ticks,
                        state["action_mask"].copy()))
            state, reward, truncated, info = await env.step(action, horizon=1500,
                                                            current_state=state)
            self.assertLessEqual(reward, 0.0)
            self.assertGreater(info["elapsed_seconds"], 0.0)
            if truncated:
                break
        return env, out

    async def test_shapes_ranges_and_signal_block(self):
        env, out = await self.episode()
        self.assertGreater(len(out), 20)
        for obs, phase, green, planned, mask in out:
            grid, aux = obs["grid"], obs["aux"]
            self.assertEqual(grid.shape, V5_GRID_SHAPE)
            self.assertEqual(aux.shape, (V5_AUX_SIZE,))
            self.assertTrue(np.isfinite(grid).all() and np.isfinite(aux).all())
            self.assertGreaterEqual(grid.min(), 0.0)
            self.assertLessEqual(grid.max(), 1.0)
            self.assertGreaterEqual(aux.min(), 0.0)
            self.assertLessEqual(aux.max(), 1.0)
            self.assertEqual(tuple(mask), (True, False, True, False))
            self.assertEqual(aux[0], float(phase == "HDT"))
            # Decisions come with 5 s of green left.
            self.assertEqual(planned - green, 25)
            self.assertAlmostEqual(aux[1], green * DT / 75.0, places=5)
            self.assertAlmostEqual(aux[3], (75.0 - planned * DT) / 60.0, places=5)
            # Vehicle channels only on existing sublanes.
            for a, arm in enumerate(ARMS):
                lanes = min(12, env.sim.roads[f"{arm}_in"]["num_lanes"])
                self.assertTrue((grid[a, 6, :, :lanes] == 1).all())
                self.assertEqual(grid[a, :6, :, lanes:].sum(), 0.0)
                # Kind flags imply a vehicle; stopped implies a vehicle.
                for k in (1, 2, 3, 4):
                    self.assertTrue((grid[a, k] <= grid[a, 0]).all())

    async def test_queue_features_follow_the_grid(self):
        env, out = await self.episode(decisions=30)
        busy = 0
        for obs, *_ in out:
            grid, aux = obs["grid"], obs["aux"]
            for a in range(4):
                if aux[4 + a] > 0:
                    busy += 1
                    self.assertGreater(grid[a, 4].sum(), 0.0)
                if grid[a, 4].sum() == 0:
                    self.assertEqual(aux[4 + a], 0.0)
                    self.assertEqual(aux[20 + a], 0.0)
                edge = grid[a, 0, 8:10][:, grid[a, 6, 0] > 0].mean()
                self.assertAlmostEqual(aux[12 + a], edge, places=5)
        self.assertGreater(busy, 10)
        # After a few decisions the arrival EWMA is positive on every arm.
        self.assertTrue((out[-1][0]["aux"][16:20] > 0).all())

    async def test_reset_clears_the_rate_memory(self):
        env, _ = await self.episode(decisions=10)
        state = await env.reset(seed=12, parameters=self.parameters)
        obs = featurize(state)
        self.assertTrue((obs["aux"][16:20] == 0).all())
        self.assertTrue((obs["aux"][28:30] == 0).all())

    def test_v5_needs_the_spatial_state(self):
        with self.assertRaises(ValueError):
            LVLPythonEnv(state_mode="dtse", reward_mode="reference", state_version="v5")
        with self.assertRaises(ValueError):
            LVLPythonEnv(state_mode="spatial", reward_mode="r1", state_version="v6")


def fake_obs(rng):
    grid = (rng.random(V5_GRID_SHAPE) < .2).astype(np.float32)
    grid[:, 5] *= rng.random(V5_GRID_SHAPE[1:])[0]
    aux = rng.random(V5_AUX_SIZE).astype(np.float32)
    return {"grid": grid, "aux": aux}


class RainbowV5Tests(unittest.TestCase):
    def test_pack_roundtrip(self):
        rng = np.random.default_rng(0)
        obs = fake_obs(rng)
        blob, aux = ra.pack_observation(obs, ra.GRID_SHAPE_V5)
        grid, restored = ra.unpack_observations([blob], [aux], ra.GRID_SHAPE_V5)
        self.assertLessEqual(np.abs(grid[0] - obs["grid"]).max(), 1 / 255 + 1e-6)
        self.assertLessEqual(np.abs(restored[0] - obs["aux"]).max(), 1e-3)
        bad = dict(obs, aux=obs["aux"] + 2.0)
        with self.assertRaises(ValueError):
            ra.pack_observation(bad, ra.GRID_SHAPE_V5)

    def test_agent_acts_learns_saves_and_rejects_v4(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(1)
        agent = RainbowAgent(buffer_size=64, warmup=8, batch_size=4, atoms=11,
                             v_min=-100.0, v_max=0.0, reward_mode="r4", device="cpu",
                             reward_settings={"r1_ref": 50.0, "state_version": "v5"})
        self.assertEqual(agent.architecture, ra.ARCHITECTURE_V5)
        self.assertEqual(agent.aux_size, 30)
        mask = np.asarray([True, False, True, False])
        for i in range(20):
            s, s2 = fake_obs(rng), fake_obs(rng)
            a = agent.act(s, action_mask=mask)
            self.assertIn(a, (0, 2))
            agent.store(s, a, -float(rng.random()), s2, False,
                        seconds=5.0 if i % 3 else 19.0, next_mask=mask)
        agent.finish_episode()
        self.assertTrue(agent.can_update())
        before = [p.detach().clone() for p in agent.online_net.parameters()]
        for _ in range(3):
            self.assertTrue(np.isfinite(agent.update()))
        self.assertTrue(any((p != q).any() for p, q in zip(before, agent.online_net.parameters())))
        # SMDP discount: each 3-step window holds one 19 s and two 5 s
        # transitions, so it is discounted by gamma ** (29 / 5).
        discounts = agent.buffer.discounts[:agent.buffer.size - 2]
        self.assertTrue(np.allclose(discounts, .99 ** (29 / 5), atol=1e-6))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.pt"
            agent.save(path)
            other = RainbowAgent(buffer_size=1, warmup=1, batch_size=1, atoms=11,
                                 v_min=-100.0, v_max=0.0, reward_mode="r4", device="cpu",
                                 reward_settings={"r1_ref": 50.0, "state_version": "v5"})
            other.load(path)
            obs = fake_obs(rng)
            self.assertEqual(other.act(obs, greedy=True, action_mask=mask),
                             agent.act(obs, greedy=True, action_mask=mask))
            legacy = RainbowAgent(buffer_size=1, warmup=1, batch_size=1, atoms=11,
                                  v_min=-100.0, v_max=0.0, reward_mode="r4", device="cpu")
            with self.assertRaises(ValueError):
                legacy.load(path)
            state = agent.training_state()
            resumed = RainbowAgent(buffer_size=64, warmup=8, batch_size=4, atoms=11,
                                   v_min=-100.0, v_max=0.0, reward_mode="r4", device="cpu",
                                   reward_settings={"r1_ref": 50.0, "state_version": "v5"})
            resumed.restore_training(state)
            self.assertEqual(len(resumed.buffer), len(agent.buffer))
            self.assertTrue(np.isfinite(resumed.update()))

    def test_v4_default_is_unchanged(self):
        agent = RainbowAgent(buffer_size=1, warmup=1, batch_size=1, device="cpu")
        self.assertEqual(agent.architecture, ra.ARCHITECTURE)
        self.assertEqual(agent.grid_shape, ra.GRID_SHAPE)
        self.assertEqual(agent.aux_size, 17)


class TrainArgsV5Tests(unittest.TestCase):
    def test_flag_travels_in_reward_settings_and_is_validated(self):
        from train.logic import train
        args = train.parse_args(["--algorithm", "rainbow", "--backend", "python",
                                 "--scenario", "lvl", "--reward-mode", "r4",
                                 "--state-version", "v5"])
        self.assertEqual(train.reward_settings(args)["state_version"], "v5")
        self.assertEqual(train.training_config(args)["state_version"], "v5")
        with self.assertRaises(SystemExit):
            train.parse_args(["--algorithm", "hybrid", "--backend", "python",
                              "--scenario", "lvl", "--reward-mode", "r4",
                              "--state-version", "v5"])


if __name__ == "__main__":
    unittest.main()
