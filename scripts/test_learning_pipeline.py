"""Regression checks for clearance, time-limit censoring, action masks and cache identity."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.fair_benchmark import EXTRA_FIELDS, Learners, run_episode
from scripts.benchmark_3dqn_vs_4 import run_three_episode, run_job, evaluation_signature
from train.logic.evaluate import evaluate_episode
from train.logic.gama_env import GamaTrafficEnv
from train.logic.three_dqn_agent import ThreeDQNAgent


class ClearanceEnv:
    """A 22-second action followed by a horizon cutting the next action short."""
    def __init__(self, three=True):
        self.three = three
        self.cycle = 0
        self._server_messages = []
        self.reward_mode = "queue_time"
        self.output = None
        self.actions = []

    async def reset(self, **kwargs):
        self.cycle = 0
        return await self.read_state()

    async def reset_demand(self, seed, path):
        return await self.reset()

    async def _exec_expression(self, expression):
        if expression == "world.seed":
            return 5001
        fields = [p.strip() for p in expression.replace('"" + ', '').split(' + "|" + ')]
        state = await self.read_state()
        return '|'.join(str(state.get(k, 0.0)) for k in fields)

    async def read_state(self, force_dtse=False):
        waiting = self.cycle in (100, 320, 600)
        next_tick = next((x for x in (100, 320, 600) if x > self.cycle), 800)
        # Before the first 15-second estimate expires, GAMA can only give a
        # lower bound. Clearance extends the true next decision to cycle 320.
        scheduled = min(next_tick, 251) if 100 < self.cycle < 251 else next_tick
        state = dict.fromkeys(EXTRA_FIELDS, 0.0)
        state.update(cycle=self.cycle, dqn_waiting_action=waiting,
                     next_decision_in_cycles=1 if waiting else scheduled-self.cycle,
                     global_phase=0 if self.cycle <= 100 else 4,
                     green_cycles=100, queue_time_integral=self.cycle * .2,
                     total_vehicles_created=0, total_vehicles_passed=0,
                     current_queue_length=2, average_waiting_time=0,
                     average_speed=0, average_travel_time=0, max_waiting_time=0)
        if self.three and (waiting or force_dtse):
            state.update(dtse=np.zeros((3, 8, 20), dtype=np.float32), delay_sq=2.,
                         action_mask=np.array([False, True, False, False]) if self.cycle==100
                         else np.array([False, False, True, False]))
        return state

    async def set_action(self, action):
        self.actions.append((self.cycle, action))

    async def step_cycles(self, count):
        assert count > 0
        self.cycle += count

    transition_reward = GamaTrafficEnv.transition_reward


class SampleWriter:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    def agent(self):
        return ThreeDQNAgent(buffer_size=4, warmup=2, batch_size=2,
                             device=torch.device('cpu'), reward_mode='queue_time')

    async def test_evaluation_handles_extended_clearance_and_final_reward(self):
        env, writer = ClearanceEnv(), SampleWriter()
        state, reward, decisions, count = await evaluate_episode(env, self.agent(), 5001, 400, writer, 1)
        self.assertEqual(state['cycle'], 400)
        self.assertEqual(env.actions, [(100, 1), (320, 2)])
        self.assertEqual(decisions, 2)
        self.assertAlmostEqual(reward, -1.2)
        self.assertEqual(count, 39)
        self.assertEqual([r['cycle'] for r in writer.rows], list(range(10, 400, 10)))

    async def test_validation_skips_unneeded_sample_reads(self):
        env, writer = ClearanceEnv(), SampleWriter()
        state, reward, decisions, count = await evaluate_episode(
            env, self.agent(), 5001, 400, writer, 1, collect_samples=False)
        self.assertEqual((state['cycle'], decisions, count), (400, 2, 0))
        self.assertEqual(writer.rows, [])
        self.assertAlmostEqual(reward, -1.2)

    async def test_all_learners_drop_partial_action_instead_of_false_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demand = root / 'demand.csv'
            demand.write_text('0,0,0,30,0,4\n')
            for method in ('Q-Learning', 'Deep Q-Learning'):
                out = root / method
                metrics = await run_episode(ClearanceEnv(False), Learners(7), method,
                    5001, 400, demand, out, train=True, collect_only=True)
                transitions = json.loads((out / 'transitions.json').read_text())
                self.assertEqual(len(transitions), 1)
                self.assertFalse(transitions[0][4])
                self.assertEqual(transitions[0][5], 22.0)
                self.assertAlmostEqual(transitions[0][2], -.88)
                self.assertAlmostEqual(metrics['reward'], -1.2)
                self.assertTrue(metrics['truncated_transition'])
            metrics, transitions = await run_three_episode(ClearanceEnv(), self.agent(),
                5001, 400, demand, root/'3dqn', train=True)
            self.assertEqual(len(transitions), 1)
            self.assertFalse(transitions[0][4])
            self.assertEqual(transitions[0][5], 22.0)
            np.testing.assert_array_equal(transitions[0][6], [False, False, True, False])
            self.assertAlmostEqual(transitions[0][2], -.88)
            self.assertAlmostEqual(metrics['reward'], -1.2)

    async def test_changed_baseline_checkpoint_invalidates_completed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demand = root/'demand.csv'; demand.write_text('0,0,0,30,0,4\n')
            for method, filename in [('Q-Learning', 'best_q.pt'), ('Deep Q-Learning', 'best_dqn.pt')]:
                checkpoint = root/filename; checkpoint.write_bytes(b'old')
                spec = dict(method=method, seed=5001, train=False, label='cache',
                            folder=method, demand=demand)
                out = root/method; out.mkdir()
                from scripts.benchmark_3dqn_vs_4 import file_hash
                (out/'metrics.json').write_text(json.dumps(dict(method=method, seed=5001,
                    horizon=400, demand_sha256=file_hash(demand),
                    evaluation_sha256=evaluation_signature(spec, root, None, 400))))
                for name in ('series.csv', 'decisions.csv'):
                    (out/name).write_text('complete\n')
                saved, _ = await run_job(spec, 0, root, root, None, 400)
                self.assertEqual(saved['method'], method)
                checkpoint.write_bytes(b'new')
                with patch('scripts.benchmark_3dqn_vs_4.GamaServerManager') as manager:
                    manager.return_value.start.side_effect = RuntimeError('fresh evaluation required')
                    with self.assertRaisesRegex(RuntimeError, 'fresh evaluation required'):
                        await run_job(spec, 0, root, root, None, 400)
