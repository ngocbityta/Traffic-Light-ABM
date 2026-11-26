#!/usr/bin/env python3
"""Train four-phase Q-learning, DQN, 3DQN, Rainbow or Hybrid with GAMA demand.

Each CPU worker runs an independent GAMA simulation. A single learner merges
transitions in seed order, updates the policy, and saves full resume state.
"""

import argparse
import asyncio
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import pickle
import queue
import random
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .three_dqn_agent import ThreeDQNAgent, FlatDQNAgent
from .rainbow_agent import RainbowAgent
from .hybrid_agent import SpatialHybridAgent
from .tabular_agent import TabularQAgent
from .gama_env import GamaTrafficEnv, QUEUE_FEATURE_SCALES, featurize
from .python_env import PythonTrafficEnv
from .lvl_env import LVLGamaEnv, LVLPythonEnv
from .server_manager import GamaServerManager
from .evaluate import evaluate_episode
from .demand import generate_demand, demand_parameters
from python_simulation.demand import (axis_flows, generate_demand as generate_lvl_demand,
                                      sample_demand_level)
from python_simulation.demand import demand_parameters as lvl_demand_parameters
from python_simulation.sim_config import load_sim_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train traffic agents on Le Van Luong or the legacy intersection")
    p.add_argument("--scenario", choices=("lvl", "legacy"), default="lvl",
                   help="Le Van Luong two-phase geometry is the current training target")
    p.add_argument("--algorithm", choices=("qlearning", "dqn", "3dqn", "rainbow", "hybrid"), default="3dqn")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume-dir", type=str, default=None,
                   help="Resume from training_state.pt with replay and optimizer intact")
    p.add_argument("--episodes", type=int, default=0,
                   help="Optional rollout limit (0 = follow total-steps); multiworker rollouts may be partial episodes")
    p.add_argument("--total-steps", type=int, default=1_000_000,
                   help="Agent decisions; use 1000000 or 5000000 as in the paper")
    p.add_argument("--max-train-seconds", type=float, default=None,
                   help="Stop after this much training wall time, at a decision boundary")
    p.add_argument("--episode-horizon", type=int, default=8000,
                   help="So cycle GAMA moi episode (8000 cycle = 800s, giong batch cac model khac)")
    p.add_argument("--port", type=int, default=6868)
    p.add_argument("--backend", choices=("gama", "python"), default="python",
                   help="Simulation engine; compare both on the selected scenario")
    p.add_argument("--python-sim-mode", choices=("safe", "gama_2025_6_4"),
                   default="safe",
                   help="LVL Python backend only: safe (default) or GAMA 2025.6.4 parity mode; ignored on GAMA backend")
    p.add_argument("--sim-config", type=str, default=None,
                   help="Shared simulator JSON or calibrate_discharge output")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--start-server", action="store_true",
                   help="Tu khoi dong gama-server (neu chua chay)")
    p.add_argument("--workers", type=int, default=1,
                   help="Independent simulation workers; one learner merges batches in seed order")
    p.add_argument("--rollout-mode", choices=("async", "sync"), default="async",
                   help="Python workers: async = each worker gets a new rollout as soon as "
                        "its last one is learned (no waiting for the slowest worker); "
                        "sync = the previous lock-step rounds (reproducible order)")
    p.add_argument("--replay-ratio", type=float, default=1.0,
                   help="Gradient updates per stored transition (1.0 = previous behaviour)")
    p.add_argument("--learner", choices=("thread", "inline"), default="thread",
                   help="async Python workers: 'thread' learns in a background thread so "
                        "finished workers get their next rollout at once; 'inline' learns "
                        "in the dispatch loop before resubmitting (previous behaviour)")
    p.add_argument("--max-learner-lag", type=int, default=None,
                   help="Rollouts that may wait for the learner thread before new rollouts "
                        "are held back (keeps the replay ratio; default: number of workers)")
    p.add_argument("--learner-threads", type=int, default=None,
                   help="torch threads of the learner (default: free cores beside the workers)")
    p.add_argument("--rollout-decisions", type=int, default=16,
                   help="Decisions per worker before refreshing the shared policy (multiworker only)")
    p.add_argument("--gama-heap-mb", type=int, default=2048,
                   help="Maximum Java heap per GAMA worker; default 2048 MB")
    p.add_argument("--fast-simulation", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Omit presentation and reward metrics unused by training (default: on)")
    p.add_argument("--keep-server", action="store_true",
                   help="Khong dung server sau khi train xong")
    p.add_argument("--seed-base", type=int, default=1000,
                   help="Seed GAMA cua episode e = seed_base + e")
    p.add_argument("--validation-seed-base", type=int, default=10_000_000,
                   help="Reserved held-out seed range; validation uses base+1, base+2, ...")
    # Demand level (lvl scenario). Each training episode draws its own level
    # from its seed and keeps it for the whole episode: the mean flow of the
    # four arms is scale * 2400 veh/h and every arm of the Hoang Dao Thuy
    # axis (NW-SE) carries 1 + imbalance times the flow of a Le Van Luong arm.
    p.add_argument("--demand-scale-min", type=float, default=1.0,
                   help="Lowest demand level of a training episode (1.0 = 2400 veh/h per arm)")
    p.add_argument("--demand-scale-max", type=float, default=1.0,
                   help="Highest demand level of a training episode")
    p.add_argument("--axis-imbalance-max", type=float, default=0.0,
                   help="Largest extra share of the NW-SE axis per arm (0.2 = 20 %% busier)")
    p.add_argument("--queue-feature-scale", choices=QUEUE_FEATURE_SCALES, default="linear",
                   help="Queue counts in the observation: linear (/100, /50, clipped) or "
                        "log (log1p(n)/log1p(1000); does not saturate at heavy demand)")
    p.add_argument("--validation-levels", default=None,
                   help="Held-out levels 'scale:imbalance,...', one seed each "
                        "(default: two seeds at 1:0; when the training level is random, the "
                        "lowest, middle and highest scale at half the largest imbalance)")
    # DQN hyperparameters
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--reward-mode", choices=("reference", "queue_time", "total_wait", "r1", "r2", "r3", "r4"), default=None)
    p.add_argument("--r1-ref", type=float, default=50.0,
                   help="Fixed D_ref for R1/R3/R4; calibrate against fixed-time before final experiments")
    p.add_argument("--r2-ref", type=float, default=50.0, help="Fixed W_ref for R2")
    p.add_argument("--b-ref", type=float, default=50.0, help="Fixed B_ref for R3")
    p.add_argument("--r2-lambda", type=float, default=1.0, help="Weight of external pending vehicles in R2")
    p.add_argument("--r3-beta", type=float, default=1.0, help="Weight of blocked outgoing links in R3")
    p.add_argument("--pressure-lambda", type=float, default=.5,
                   help="Hybrid score = normalized Q + lambda * normalized pressure")
    p.add_argument("--demand-mode", choices=("paired", "model"), default="paired",
                   help="Paired CSV requests keep arrivals identical across policies; model uses live GAMA spawning")
    p.add_argument("--max-delay-sq", type=float, default=None,
                   help="Optional fixed tsd_max; default follows the paper's running maximum")
    p.add_argument("--validate-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tau", type=float, default=1e-3)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-decay-steps", type=float, default=None,
                   help="Time constant of the exponential epsilon decay; "
                        "default total_steps / 5 (epsilon ~ 0.7 %% of its start at the end)")
    p.add_argument("--eps-min", type=float, default=0.001)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--per-alpha", type=float, default=.5)
    p.add_argument("--per-beta-start", type=float, default=.4)
    p.add_argument("--per-beta-steps", type=int, default=1_000_000)
    p.add_argument("--c51-atoms", type=int, default=51)
    p.add_argument("--c51-v-min", type=float, default=-2000.0)
    p.add_argument("--c51-v-max", type=float, default=0.0)
    p.add_argument("--target-update", type=int, default=2000)
    p.add_argument("--model-path", type=str, default=None,
                   help="Optional .gaml path override for the selected scenario")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None,
                   help="Device for PyTorch learner (default: cpu on Mac, cuda if available)")
    p.add_argument("--allow-code-change", action="store_true",
                   help="Resume although the hashed source files differ from the checkpoint's "
                        "(only for deliberate code updates such as a faster learner; training "
                        "settings must still match). Later checkpoints store the new hashes.")
    p.add_argument("--fast-learner", action="store_true",
                   help="Rainbow only: same update rule with uint8 replay, one online forward "
                        "over [s; s2] and, on CUDA, the gradient step as one CUDA graph")
    args = p.parse_args(argv)
    if args.fast_learner and args.algorithm != "rainbow":
        p.error("--fast-learner is only implemented for --algorithm rainbow")
    if args.sim_config and (args.backend != "python" or args.scenario != "lvl"):
        p.error("--sim-config requires --backend python --scenario lvl")
    args.simulation_config = load_sim_config(args.sim_config)
    spatial_algorithm = args.algorithm in ("rainbow", "hybrid")
    if args.reward_mode is None:
        args.reward_mode = "r1" if spatial_algorithm else "reference"
    if spatial_algorithm != (args.reward_mode in ("r1", "r2", "r3", "r4")):
        p.error("Rainbow/hybrid require R1-R4; legacy agents require their original rewards")
    if args.buffer_size is None:
        args.buffer_size = 100_000 if spatial_algorithm else 1_000_000
    if args.warmup is None:
        args.warmup = 20_000 if spatial_algorithm else 100_000
    if (args.episode_horizon <= 100 or args.total_steps <= 0 or
            (args.max_train_seconds is not None and args.max_train_seconds <= 0) or
            args.validate_every <= 0 or args.checkpoint_every <= 0 or
            args.workers <= 0 or args.rollout_decisions <= 0 or args.gama_heap_mb < 512 or
            args.buffer_size < args.batch_size or args.warmup < 0 or
            not (args.replay_ratio > 0) or
            (args.learner_threads is not None and args.learner_threads < 1) or
            (args.max_learner_lag is not None and args.max_learner_lag < 1) or
            min(args.r1_ref, args.r2_ref, args.b_ref) <= 0 or
            min(args.r2_lambda, args.r3_beta, args.pressure_lambda) < 0):
        p.error("Horizon must exceed 100 cycles; total-steps and validate-every must be positive")
    if args.backend == "python" and args.start_server:
        p.error("--start-server is only valid with --backend gama")
    if args.python_sim_mode != "safe" and (args.backend != "python" or args.scenario != "lvl"):
        p.error("--python-sim-mode gama_2025_6_4 requires --backend python --scenario lvl")
    if args.backend == "python" and args.demand_mode != "paired":
        p.error("Python backend requires paired CSV demand; GAMA's live RNG is different")
    if args.scenario == "lvl" and args.demand_mode != "paired":
        p.error("Le Van Luong training requires paired CSV demand")
    if args.scenario == "lvl" and args.reward_mode in ("total_wait", "r3"):
        p.error("Le Van Luong has no external source queue or blocked exit metric for this reward")
    if args.backend == "python" and args.workers > 1 and args.scenario != "lvl":
        p.error("Parallel Python rollout currently supports the Le Van Luong scenario")
    if args.backend == "python" and args.workers > 1 and args.reward_mode == "reference" and not (
            args.max_delay_sq and args.max_delay_sq > 0):
        p.error("Parallel reference reward requires a shared positive --max-delay-sq")
    if args.workers > 1 and args.backend == "gama" and not args.start_server:
        p.error("Multiple workers require --start-server for isolated GAMA processes")
    if args.workers > 1 and args.reward_mode == "reference" and not (args.max_delay_sq and args.max_delay_sq > 0):
        p.error("Parallel reference reward requires a fixed positive --max-delay-sq shared by all workers")
    if args.resume_dir and args.output_dir and Path(args.resume_dir).resolve() != Path(args.output_dir).resolve():
        p.error("--output-dir must equal --resume-dir when resuming")
    levels = (args.demand_scale_min, args.demand_scale_max, args.axis_imbalance_max)
    if not all(math.isfinite(x) for x in levels) or not (
            0 < args.demand_scale_min <= args.demand_scale_max and args.axis_imbalance_max >= 0):
        p.error("Demand levels need 0 < --demand-scale-min <= --demand-scale-max "
                "and --axis-imbalance-max >= 0")
    random_level = levels != (1.0, 1.0, 0.0)
    if args.validation_levels is None:
        low, high = args.demand_scale_min, args.demand_scale_max
        half = args.axis_imbalance_max / 2
        args.validation_levels = (
            f"{low!r}:{half!r},{(low + high) / 2!r}:{half!r},{high!r}:{half!r}"
            if random_level else "1:0,1:0")
    try:
        args.validation_levels = tuple(
            (float(scale), float(imbalance)) for scale, imbalance in
            (item.split(":") for item in args.validation_levels.split(",")))
    except ValueError:
        p.error("--validation-levels must look like 0.5:0.1,1:0.1,1.5:0.1")
    if not args.validation_levels or not all(
            math.isfinite(a) and math.isfinite(b) and a > 0 and b >= 0
            for a, b in args.validation_levels):
        p.error("--validation-levels needs positive scales and non-negative imbalances")
    if (random_level or set(args.validation_levels) != {(1.0, 0.0)}) and args.scenario != "lvl":
        p.error("Demand levels exist only for --scenario lvl")
    possible_episodes = min(args.total_steps, args.episodes) if args.episodes else args.total_steps
    training_end = args.seed_base + possible_episodes
    if any(args.seed_base < args.validation_seed_base + offset <= training_end
           for offset in range(1, len(args.validation_levels) + 1)):
        p.error("Validation seeds overlap the possible training seed range")
    if args.gamma is None:
        args.gamma = .97 if args.reward_mode in ("queue_time", "total_wait", "r1", "r2", "r3", "r4") else .99
    if args.lr is None:
        args.lr = 0.1 if args.algorithm == "qlearning" else 1e-4
    if args.eps_decay_steps is None:
        # eps = eps_start * exp(-decisions / eps_decay_steps). The old fixed
        # 2,000,000 left the agent 61 % random at the end of a 1M-step run.
        args.eps_decay_steps = max(1.0, args.total_steps / 5.0)
    return args


def training_config(args):
    fields = ("algorithm", "scenario", "backend", "python_sim_mode", "episode_horizon", "max_train_seconds", "validate_every", "checkpoint_every",
              "workers", "rollout_decisions", "rollout_mode", "replay_ratio", "learner", "max_learner_lag", "gama_heap_mb", "fast_simulation", "device", "seed_base",
              "validation_seed_base", "lr", "gamma",
              "reward_mode", "r1_ref", "r2_ref", "b_ref", "r2_lambda", "r3_beta",
              "pressure_lambda", "demand_mode", "demand_scale_min", "demand_scale_max",
              "axis_imbalance_max", "validation_levels", "queue_feature_scale", "max_delay_sq", "buffer_size", "warmup", "batch_size",
              "tau", "eps_start", "eps_decay_steps", "eps_min", "model_path")
    config = {name: getattr(args, name) for name in fields}
    # JSON has no tuples: keep the in-memory config equal to the saved one,
    # or a resume from training_config.json sees a changed setting.
    config["validation_levels"] = [list(level) for level in args.validation_levels]
    config["simulation_config"] = args.simulation_config
    if args.algorithm == "rainbow":
        config.update({name: getattr(args, name) for name in
                       ("n_step", "per_alpha", "per_beta_start", "per_beta_steps",
                        "c51_atoms", "c51_v_min", "c51_v_max", "target_update")})
    files = ("train/benchmarks/gama/includes/collision_avoidance.gaml",
             "train/benchmarks/gama/models/intersection-dqn-python.gaml",
             "train/logic/gama_env.py", "train/logic/python_env.py", "train/logic/evaluate.py",
             "train/logic/server_manager.py", "train/logic/demand.py",
             "train/logic/three_dqn_agent.py",
             "train/logic/tabular_agent.py", "train/logic/rainbow_agent.py", "train/logic/train.py")
    files = files + ("train/logic/hybrid_agent.py",)
    if args.scenario == "lvl":
        files += ("train/logic/lvl_env.py", "python-simulation/simulation/python_simulation/python_sim.py",
                  "python-simulation/simulation/python_simulation/python_sim.pxd",
                  "python-simulation/simulation/python_simulation/_native.py",
                  "python-simulation/simulation/python_simulation/demand.py",
                  "gama-simulation/simulation/models/LeVanLuong.gaml",
                  "python-simulation/simulation/python_simulation/includes/roads.geojson",
                  # python_sim imports these modules directly. Hash them too
                  # so a checkpoint cannot resume against changed motion,
                  # startup, or road geometry while claiming identical code.
                  "python-simulation/simulation/python_simulation/gama_2025_idm_mobil.py",
                  "python-simulation/simulation/python_simulation/gama_2025_transition.py",
                  "python-simulation/simulation/python_simulation/safe_startup.py",
                  "python-simulation/simulation/python_simulation/vehicle_geometry.py",
                  # Include safety helpers that may be imported by the safe
                  # simulator as its collision and junction logic evolves.
                  "python-simulation/simulation/python_simulation/junction_conflicts.py",
                  "python-simulation/simulation/python_simulation/junction_negotiation.py",
                  "python-simulation/simulation/python_simulation/junction_negotiation.pxd",
                  "python-simulation/simulation/python_simulation/sim_config.py",
                  "python-simulation/simulation/python_simulation/safe_collision_guard.py")
    if args.backend == "python":
        # GAMA models do not run with the Python backend. A copy of the code
        # taken to another machine (e.g. Colab) without them must still
        # resume the same checkpoint.
        files = tuple(name for name in files if not name.endswith(".gaml"))
    missing = [name for name in files if not (REPO_DIR / name).exists()]
    if args.backend == "python" and missing:
        raise FileNotFoundError("Source files needed by training are missing: " + ", ".join(missing))
    # GAMA backend: gama-simulation/ was removed on 2026-11-16; hash the files present.
    config["code_sha256"] = {name: hashlib.sha256(
        (REPO_DIR / name).read_bytes()).hexdigest() for name in files
        if (REPO_DIR / name).exists()}
    return config


# Settings that change where and how fast training runs, not what is learned:
# a run may be resumed with other values (e.g. started on a laptop, continued
# on a Colab GPU with more workers).
RESUMABLE_CHANGES = ("workers", "device", "max_train_seconds", "learner", "max_learner_lag",
                     "rollout_mode", "validate_every", "checkpoint_every", "gama_heap_mb")


class DemandManifest:
    """seed -> demand file record, written to ``demand_manifest.json`` on ``flush``.

    ``record`` only updates memory. Rewriting the whole indented JSON for every
    episode took the learner process's GIL for O(entries) time (over 100 ms at
    15k episodes) and slowed training down as the run grew, so the file is
    written at checkpoints and at the end instead: it always matches the last
    checkpoint, which is what a resume restarts from.
    """

    def __init__(self, path, entries=None):
        self.path = path
        self.entries = dict(entries or {})
        self.lock = threading.Lock()
        self.dirty = False

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            return cls(path, json.load(stream))

    def record(self, seed, path, digest, outdir, level=None):
        entry = {"path": os.path.relpath(path, outdir), "sha256": digest}
        if level is not None:
            scale, imbalance, flows = level
            entry.update(scale=round(scale, 6), axis_imbalance=round(imbalance, 6),
                         flow_veh_h={arm: round(flow, 3) for arm, flow in flows.items()})
        with self.lock:
            self.entries[str(seed)] = entry
            self.dirty = True

    def flush(self):
        with self.lock:
            if not self.dirty:
                return False
            temporary = self.path + ".tmp"
            with open(temporary, "w") as stream:
                json.dump(self.entries, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, self.path)
            self.dirty = False
            return True


def check_resume_config(saved, current, allow_code_change=False):
    """Raise unless ``saved`` and ``current`` differ only in RESUMABLE_CHANGES
    (and, with ``allow_code_change``, in the source hashes); return the keys
    that differ."""
    changed = sorted(key for key in set(saved) | set(current)
                     if saved.get(key) != current.get(key))
    allowed = RESUMABLE_CHANGES + (("code_sha256",) if allow_code_change else ())
    blocked = [key for key in changed if key not in allowed]
    if blocked:
        raise ValueError("Resume requires identical training settings and source code "
                         f"(changed: {', '.join(blocked)})")
    return changed


def actor_seed(episode_seed, learner_decisions, worker=0):
    """Distinct exploration stream for every rollout of an episode.

    An episode is collected in several rollouts. Seeding each rollout's actor
    with the bare episode seed replayed the same random action sequence in
    every one of them.
    """
    return (int(episode_seed) * 1_000_003 + int(learner_decisions) * 7919 +
            int(worker)) % (2 ** 31 - 1)


def make_actor(agent_class, learner, args, seed):
    """Freeze a small CPU policy for one rollout; never copy the learner's replay."""
    # Agent constructors seed the *global* torch and numpy generators, which
    # the learner also draws from. Building an actor must not rewind them.
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    try:
        return _make_actor(agent_class, learner, args, seed)
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(numpy_state)


def _make_actor(agent_class, learner, args, seed):
    actor = agent_class(lr=args.lr, gamma=args.gamma, buffer_size=1,
                        warmup=1, batch_size=1, tau=args.tau,
                        eps_start=args.eps_start, eps_decay_steps=args.eps_decay_steps,
                        eps_min=args.eps_min, seed=seed, reward_mode=args.reward_mode,
                        **({} if args.algorithm == "qlearning" else {"device": torch.device("cpu")}),
                        **spatial_kwargs(args))
    actor.eps = learner.eps
    actor.decisions = learner.decisions
    for name in ("q_spread", "pressure_spread"):  # hybrid score scales
        if hasattr(learner, name):
            setattr(actor, name, getattr(learner, name))
    if args.algorithm == "qlearning":
        actor.table = {key: values.copy() for key, values in learner.table.items()}
    else:
        actor.online_net.load_state_dict(learner.online_net.state_dict())
        if args.algorithm != "rainbow":
            actor.online_net.eval()
    return actor


def reward_settings(args):
    # Saved in every checkpoint and handed back to the environment by
    # evaluate.py, so the observation scale travels with the weights.
    settings = dict(r1_ref=args.r1_ref, r2_ref=args.r2_ref, b_ref=args.b_ref,
                    r2_lambda=args.r2_lambda, r3_beta=args.r3_beta)
    if args.queue_feature_scale != "linear":
        settings["queue_feature_scale"] = args.queue_feature_scale
    return settings


def spatial_kwargs(args):
    if args.algorithm == "hybrid":
        return dict(pressure_lambda=args.pressure_lambda,
                    reward_settings=reward_settings(args))
    if args.algorithm != "rainbow":
        return {}
    return dict(n_step=args.n_step, per_alpha=args.per_alpha,
                per_beta_start=args.per_beta_start, per_beta_steps=args.per_beta_steps,
                atoms=args.c51_atoms, v_min=args.c51_v_min, v_max=args.c51_v_max,
                target_update=args.target_update, r1_ref=args.r1_ref,
                reward_settings=reward_settings(args))


_UPDATE_CREDIT = [0.0]


def learn_emitted(agent, args, emitted):
    updates = emitted if args.algorithm == "rainbow" else 1
    ratio = getattr(args, "replay_ratio", 1.0)
    if ratio != 1.0:
        # Fractional replay ratio: carry the remainder between calls.
        _UPDATE_CREDIT[0] += updates * ratio
        updates = int(_UPDATE_CREDIT[0])
        _UPDATE_CREDIT[0] -= updates
    for _ in range(updates):
        if agent.can_update():
            agent.update()


async def collect_rollout(env, actor, args, seed, decision_limit, force_warmup,
                          parameters=None, deadline=None, initial_state=None):
    """Collect a bounded rollout; keep the GAMA episode alive between segments."""
    start = time.monotonic()
    if deadline is not None and deadline - start < 5.0:
        return dict(seed=seed, state=None, transitions=[], reward=0.0,
                    decisions=0, simulated_cycles=0, wall=0.0)
    if initial_state is None:
        state = await env.reset(seed=seed, parameters=parameters)
        await env.step_cycles(state["next_decision_in_cycles"])
        state = await env.read_state()
        previous_cycle = 0
    else:
        state = initial_state
        previous_cycle = state["cycle"]
    if not state["dqn_waiting_action"]:
        raise RuntimeError("GAMA did not reach the first DQN decision")
    transitions = []
    reward_sum = 0.0
    decisions = 0
    while (state["cycle"] < args.episode_horizon and decisions < decision_limit and
           (deadline is None or time.monotonic() < deadline)):
        observation = featurize(state, getattr(env, "queue_feature_scale", "linear"))
        action = actor.act(observation, action_mask=state["action_mask"],
                           force_random=force_warmup)
        next_state, reward, _, info = await env.step(
            action, horizon=args.episode_horizon, current_state=state)
        if info["bootstrap_valid"]:
            transitions.append((observation, action, reward, featurize(next_state, getattr(env, "queue_feature_scale", "linear")),
                                info["elapsed_seconds"], next_state["action_mask"].copy(),
                                not info["bootstrap_valid"]))
        actor.decay_epsilon()
        reward_sum += reward
        decisions += 1
        state = next_state
    return dict(seed=seed, state=state, transitions=transitions, reward=reward_sum,
                decisions=decisions, simulated_cycles=state["cycle"] - previous_cycle,
                wall=time.monotonic() - start)


_PYTHON_PROCESS_ENV = None
_PROCESS_ACTOR = None


def weights_payload(agent, seed):
    """What a worker needs to act like ``make_actor``: the online network's
    weights on the CPU and the actor seed. A pickled actor was 4.4 MB (two
    networks, Adam state, replay) and took ~20-30 ms to build per rollout."""
    state = {name: tensor.detach().to("cpu", copy=True)
             for name, tensor in agent.online_net.state_dict().items()}
    return ("weights", pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), int(seed))


def load_actor_payload(payload, args):
    """A pickled actor, or ``("weights", state, seed)`` loaded into an actor
    kept by this process (built once like ``make_actor``)."""
    global _PROCESS_ACTOR
    if not (isinstance(payload, tuple) and payload and payload[0] == "weights"):
        return pickle.loads(payload)
    _, state, seed = payload
    if _PROCESS_ACTOR is None:
        _PROCESS_ACTOR = RainbowAgent(
            lr=args.lr, gamma=args.gamma, buffer_size=1, warmup=1, batch_size=1,
            tau=args.tau, eps_start=args.eps_start, eps_decay_steps=args.eps_decay_steps,
            eps_min=args.eps_min, seed=seed, reward_mode=args.reward_mode,
            device=torch.device("cpu"), **spatial_kwargs(args))
    actor = _PROCESS_ACTOR
    actor.online_net.load_state_dict(pickle.loads(state))
    # As a freshly built actor: action draws and NoisyNet noise from ``seed``.
    actor.rng = random.Random(seed)
    torch.manual_seed(seed)
    return actor


class LearnerThread(threading.Thread):
    """Stores finished rollouts in the replay and runs the gradient updates
    beside the rollout dispatcher.

    Each rollout is learned exactly as the dispatch loop did it (store, the
    replay-ratio updates per stored transition, flush the n-step tail), in
    arrival order. ``lock`` is held for a whole rollout, so outside it the
    agent never has a half-flushed n-step tail. After each rollout the online
    weights are published for the next rollouts.
    """

    def __init__(self, agent, args):
        super().__init__(name="learner", daemon=True)
        self.agent, self.args = agent, args
        self.lock = threading.RLock()
        self.items = queue.Queue()
        self.pending = 0            # rollouts queued or being learned
        self.changed = threading.Condition()
        self.error = None
        self.busy_seconds = 0.0
        self.weights = None
        self.publish()

    def publish(self):
        if self.args.algorithm == "rainbow":
            self.weights = weights_payload(self.agent, 0)[1]

    def submit(self, transitions):
        with self.changed:
            self.pending += 1
        self.items.put(transitions)

    def stop(self):
        self.items.put(None)

    def run(self):
        agent, args = self.agent, self.args
        while True:
            transitions = self.items.get()
            if transitions is None:
                return
            started = time.monotonic()
            try:
                with self.lock:
                    for observation, action, reward, next_observation, seconds, mask, terminal in transitions:
                        emitted = agent.store(observation, action, reward, next_observation,
                                              terminal, seconds=seconds, next_mask=mask)
                        learn_emitted(agent, args, emitted)
                    if args.algorithm == "rainbow":
                        learn_emitted(agent, args, agent.finish_episode())
                    self.publish()
            except BaseException as error:  # surfaced by the dispatcher
                self.error = error
            finally:
                self.busy_seconds += time.monotonic() - started
                with self.changed:
                    self.pending -= 1
                    self.changed.notify_all()
            if self.error is not None:
                return

    def check(self):
        if self.error is not None:
            raise RuntimeError("Learner thread failed") from self.error

    async def wait_below(self, limit):
        """Wait (without blocking the event loop) until fewer than ``limit``
        rollouts wait for the learner; ``limit=1`` drains it."""
        waited = time.monotonic()
        while True:
            self.check()
            with self.changed:
                if self.pending < limit:
                    return time.monotonic() - waited
            await asyncio.sleep(0.002)


def collect_python_rollout_process(actor_payload, args, seed, decision_limit,
                                   force_warmup, parameters, deadline, initial_state,
                                   sim_payload):
    """Keep one Python simulation alive in each dedicated worker process."""
    global _PYTHON_PROCESS_ENV
    if _PYTHON_PROCESS_ENV is None:
        # The actor evaluates one small observation per decision; torch's
        # default (one thread per core) in every worker oversubscribes the CPU.
        torch.set_num_threads(1)
        _PYTHON_PROCESS_ENV = LVLPythonEnv(
            model_path=args.model_path, reward_mode=args.reward_mode,
            python_sim_mode=args.python_sim_mode,
            sim_config=args.simulation_config,
            max_delay_sq=args.max_delay_sq,
            state_mode="spatial" if args.algorithm in ("rainbow", "hybrid") else "dtse",
            reset_mode="fresh", fast_training=args.fast_simulation,
            **reward_settings(args))
    if initial_state is not None and _PYTHON_PROCESS_ENV.sim is None:
        if sim_payload is None:
            raise RuntimeError("Missing Python simulation state for a resumed rollout")
        _PYTHON_PROCESS_ENV.sim = pickle.loads(sim_payload)
    actor = load_actor_payload(actor_payload, args)
    result = asyncio.run(collect_rollout(
        _PYTHON_PROCESS_ENV, actor, args, seed, decision_limit, force_warmup,
        parameters=parameters, deadline=deadline, initial_state=initial_state))
    result["sim_payload"] = (None if result["state"] is None or result["state"]["cycle"] >= args.episode_horizon
                             else pickle.dumps(_PYTHON_PROCESS_ENV.sim,
                                               protocol=pickle.HIGHEST_PROTOCOL))
    return result

def validation_score(env, state, horizon, default_dt):
    """Mean queue/cost per simulated second of one finished validation episode."""
    wait = (env.reward_cost_from_state(state) if env.reward_mode in ("r1", "r2", "r3", "r4")
            else state["queue_time_integral"])
    if env.reward_mode == "total_wait":
        wait += state["external_wait_integral"]
    return wait / (horizon * getattr(env, "step_seconds", default_dt))


_VALIDATION_PROCESS_ENV = None


def validate_seed_process(env_bytes, actor_bytes, seed, horizon, parameters, default_dt):
    """Score one frozen policy on one held-out seed in a dedicated validation process.

    The process owns its own simulation environment, so validation shares
    nothing with the learner or with the rollout workers.
    """
    global _VALIDATION_PROCESS_ENV
    if _VALIDATION_PROCESS_ENV is None:
        # One small observation per decision: extra torch threads only oversubscribe the CPU.
        torch.set_num_threads(1)
        _VALIDATION_PROCESS_ENV = pickle.loads(env_bytes)
    env = _VALIDATION_PROCESS_ENV
    actor = pickle.loads(actor_bytes)
    state, _, _, _ = asyncio.run(evaluate_episode(
        env, actor, seed, horizon, None, 0, collect_samples=False, parameters=parameters))
    return validation_score(env, state, horizon, default_dt)


async def amain(args):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.resume_dir or args.output_dir or os.path.join(
        REPO_DIR, "train", "results", f"{args.algorithm}_{stamp}")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    print(f"[Train] Ket qua se luu tai: {outdir}")
    if args.backend == "python":
        print("[Train] Python/GAMA parity is measured separately; vehicle trajectories may differ")
    config = training_config(args)
    config_path = os.path.join(outdir, "training_config.json")
    if args.resume_dir and os.path.exists(config_path):
        with open(config_path) as stream:
            changed = check_resume_config(json.load(stream), config, args.allow_code_change)
        if changed:
            print("[Train] Resuming with changed run settings: " + ", ".join(
                f"{key}" for key in changed))
    if not (args.resume_dir and os.path.exists(config_path)) or changed:
        with open(config_path, "w") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
    demand_manifest_path = os.path.join(outdir, "demand_manifest.json")
    if args.resume_dir and os.path.exists(demand_manifest_path):
        manifest = DemandManifest.load(demand_manifest_path)
    else:
        manifest = DemandManifest(demand_manifest_path)
    demand_manifest = manifest.entries  # the same dict, for the training summary

    def record_demand(seed, path, digest, level=None):
        # Memory only: the file is written by manifest.flush() at checkpoints.
        manifest.record(seed, path, digest, outdir, level)

    validation_levels = {args.validation_seed_base + 1 + i: level
                         for i, level in enumerate(args.validation_levels)}

    def demand_level(seed):
        """(scale, imbalance) of an episode: fixed for held-out seeds, else drawn from the seed."""
        if seed in validation_levels:
            return validation_levels[seed]
        return sample_demand_level(seed, args.demand_scale_min, args.demand_scale_max,
                                   args.axis_imbalance_max)

    def episode_parameters(seed):
        if args.demand_mode == "model":
            return None
        path = Path(outdir) / "demand" / f"seed_{seed}_h{args.episode_horizon}.csv"
        if args.scenario == "lvl":
            # The GAMA model has no heavy vehicles: its demand stays two-type.
            heavy_ratio = (args.simulation_config["heavy_ratio"]
                           if args.backend == "python" else 0.0)
            scale, imbalance = demand_level(seed)
            flows = axis_flows(scale, imbalance)
            digest = generate_lvl_demand(path, seed, args.episode_horizon,
                                         heavy_ratio=heavy_ratio, flow_by_arm=flows)
            record_demand(seed, path, digest, (scale, imbalance, flows))
            return lvl_demand_parameters(path)
        digest = generate_demand(path, seed, args.episode_horizon)
        record_demand(seed, path, digest)
        return demand_parameters(path)

    # 1. Server
    managers = []
    envs = []
    process_executors = []
    validation_executors = []
    try:
        if args.backend == "python" and args.workers > 1:
            # "spawn": forking a process that already holds torch threads
            # deadlocks on Linux/Colab.
            spawn = multiprocessing.get_context("spawn")
            process_executors = [concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=spawn) for _ in range(args.workers)]
        if args.start_server:
            for index in range(args.workers):
                port = args.port + index
                manager = GamaServerManager(
                    port=port, workdir=os.path.join(outdir, "server_ws", str(port)),
                    log_file=os.path.join(outdir, f"server_{port}.log"),
                    heap_mb=args.gama_heap_mb)
                manager.start(reuse_running=False)
                managers.append(manager)
        # 2. Environment + Agent
        for index in range(args.workers):
            if args.scenario == "lvl":
                env_type = LVLPythonEnv if args.backend == "python" else LVLGamaEnv
            else:
                env_type = PythonTrafficEnv if args.backend == "python" else GamaTrafficEnv
            env = env_type(host=args.host, port=args.port + index,
                                 model_path=args.model_path, reward_mode=args.reward_mode,
                                 max_delay_sq=args.max_delay_sq,
                                 state_mode="spatial" if args.algorithm in ("rainbow", "hybrid") else "dtse",
                                 reset_mode="fresh" if args.workers > 1 else "reload",
                                 fast_training=args.fast_simulation,
                                 **({"python_sim_mode": args.python_sim_mode}
                                    if args.scenario == "lvl" and args.backend == "python" else {}),
                                 **({"sim_config": args.simulation_config}
                                    if args.scenario == "lvl" and args.backend == "python" else {}),
                                 **reward_settings(args))
            await env.setup()
            envs.append(env)
        env = envs[0]
        validation_seeds = tuple(sorted(validation_levels))
        validation_dt = getattr(env, "step_seconds", .2 if args.scenario == "lvl" else .1)
        validation_parameters = {}
        validation_env_bytes = None
        if args.backend == "python":
            # Validation runs on its own two processes (one per held-out seed),
            # so neither the learner nor the rollout workers ever wait for it.
            # Held-out seeds are fixed: their demand is generated once, here.
            validation_parameters = {seed: episode_parameters(seed)
                                     for seed in validation_seeds}
            validation_env_bytes = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
            spawn = multiprocessing.get_context("spawn")
            validation_executors = [concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=spawn) for _ in validation_seeds]
        agent_class = {"qlearning": TabularQAgent, "dqn": FlatDQNAgent,
                       "3dqn": ThreeDQNAgent, "rainbow": RainbowAgent,
                       "hybrid": SpatialHybridAgent}[args.algorithm]
        agent_kwargs = spatial_kwargs(args)
        if args.algorithm != "qlearning" and args.device is not None:
            agent_kwargs["device"] = torch.device(args.device)
        agent = agent_class(
            lr=args.lr, gamma=args.gamma, buffer_size=args.buffer_size,
            warmup=args.warmup, batch_size=args.batch_size, tau=args.tau,
            eps_start=args.eps_start, eps_decay_steps=args.eps_decay_steps,
            eps_min=args.eps_min, seed=args.seed_base, reward_mode=args.reward_mode,
            **agent_kwargs)
        if args.fast_learner:
            agent.enable_fast_learner()
            print(f"[Train] fast learner on (CUDA graph: {'yes' if agent.use_graph else 'no'})")
        state_label = "4x6x20x16 + 17 aux" if args.algorithm in ("rainbow", "hybrid") else "3x8x20"
        print(f"[Train] {args.algorithm} | Scenario: {args.scenario} | Device: {agent.device} | State: {state_label} | Actions: {'2 legal phases' if args.scenario == 'lvl' else '4 phases'}")

        # 3. Vong lap huan luyen
        log_path = os.path.join(outdir, "training_log.csv")
        header = ["rollout", "seed", "final_cycle", "decisions", "rollout_reward", "epsilon",
                  "avg_loss", "updates", "throughput", "vehicles_created",
                  "avg_waiting_time", "max_waiting_time", "avg_speed",
                  "avg_travel_time", "final_queue", "wall_seconds",
                  "simulated_cycles", "completed_episode"]
        best_validation_queue = float("inf")
        class DiscardSamples:
            def writerow(self, row):
                pass
        ep = 0             # log rows: rollouts with workers > 1, episodes otherwise
        episodes_done = 0  # completed episodes: validation/checkpoint cadence
        next_seed = args.seed_base + 1
        worker_states = [None] * args.workers
        worker_seeds = [None] * args.workers
        worker_sim_payloads = [None] * args.workers
        validation_queue = deque()  # frozen-policy jobs in submission order
        validation_log_path = os.path.join(outdir, "validation_log.csv")

        def record_validation(episode, step, scores, weights=None):
            """Log one finished validation and keep the best policy; returns a marker."""
            nonlocal best_validation_queue
            if len(set(args.validation_levels)) > 1:
                # Levels differ in size: a geometric mean weighs the same
                # relative gain at every level equally, instead of letting
                # the heaviest level decide which policy is best.
                score = math.exp(sum(math.log(max(x, 1e-12)) for x in scores) / len(scores))
            else:
                score = sum(scores) / len(scores)
            marker = ""
            if score < best_validation_queue:
                best_validation_queue = score
                path = Path(outdir, "best_weights.pt")
                if weights is None:
                    agent.save(str(path))
                else:
                    # The policy that was scored, not the learner's current weights.
                    tmp = path.with_suffix(".pt.tmp")
                    tmp.write_bytes(weights)
                    os.replace(tmp, path)
                marker = "  <- best validation queue"
            with open(validation_log_path, "a", newline="") as f:
                csv.writer(f).writerow([episode, step, *scores, score])
            if weights is not None:
                print(f"[Validation] episode {episode} | step {step} | "
                      f"score {score:.4f}{marker}", flush=True)
            return marker

        def submit_validation(job):
            loop = asyncio.get_running_loop()
            job["futures"] = [
                loop.run_in_executor(validation_executors[i], validate_seed_process,
                                     validation_env_bytes, job["actor"], seed,
                                     args.episode_horizon, validation_parameters[seed],
                                     validation_dt)
                for i, seed in enumerate(validation_seeds)]
            validation_queue.append(job)

        async def collect_validation(wait=False):
            """Record finished validations in order; block only when ``wait``."""
            while validation_queue:
                job = validation_queue[0]
                if not all(future.done() for future in job["futures"]):
                    if not wait:
                        return
                    await asyncio.wait(job["futures"])
                scores = [future.result() for future in job["futures"]]
                validation_queue.popleft()
                record_validation(job["episode"], job["step"], scores, job["weights"])

        async def validate_policy(episode, reset_worker0=False):
            """Score the current policy on the held-out seeds.

            Python backend: freeze it and hand it to the validation processes
            without waiting. GAMA backend (no spare simulation process): run it
            here and return the best-so-far marker.
            """
            if validation_executors:
                weights = io.BytesIO()
                agent.save(weights)
                actor = make_actor(agent_class, agent, args, args.validation_seed_base)
                submit_validation(dict(
                    episode=episode, step=agent.decisions, weights=weights.getvalue(),
                    actor=pickle.dumps(actor, protocol=pickle.HIGHEST_PROTOCOL)))
                return ""
            # Validation must not change the training reward normalizer.
            normalizer = env._worst_delay_sq
            scores = []
            try:
                for validation_seed in validation_seeds:
                    last, _, _, _ = await evaluate_episode(
                        env, agent, validation_seed, args.episode_horizon,
                        DiscardSamples(), 0, collect_samples=False,
                        parameters=episode_parameters(validation_seed))
                    scores.append(validation_score(env, last, args.episode_horizon,
                                                   validation_dt))
            finally:
                env._worst_delay_sq = normalizer
                if reset_worker0:
                    # Validation ran on worker 0's environment.
                    worker_states[0] = None
                    worker_seeds[0] = None
            return record_validation(episode, agent.decisions, scores)

        if args.resume_dir:
            snapshot = torch.load(os.path.join(outdir, "training_state.pt"),
                                  map_location=agent.device, weights_only=False)
            check_resume_config(snapshot["config"], config, args.allow_code_change)
            agent.restore_training(snapshot["agent"])
            ep = snapshot["episode"]
            next_seed = snapshot["next_seed"]
            best_validation_queue = snapshot["best_validation_queue"]
            pending_validations = snapshot.get("pending_validations", [])
            if pending_validations and "best_before_validation" in snapshot:
                # A validation newer than this checkpoint may have replaced the file.
                best_path = Path(outdir, "best_weights.pt")
                tmp = best_path.with_suffix(".pt.tmp")
                tmp.write_bytes(snapshot["best_before_validation"])
                os.replace(tmp, best_path)
            if process_executors and "worker_states" in snapshot:
                # Another worker count: unfinished episodes beyond it are
                # dropped (their transitions are already in the replay),
                # extra workers start new episodes.
                count = args.workers
                worker_states = (list(snapshot["worker_states"]) + [None] * count)[:count]
                worker_seeds = (list(snapshot["worker_seeds"]) + [None] * count)[:count]
                worker_sim_payloads = (list(snapshot["worker_sim_payloads"]) + [None] * count)[:count]
                dropped = sum(state is not None for state in snapshot["worker_states"][count:])
                if dropped:
                    print(f"[Train] {dropped} unfinished episode(s) of removed workers dropped")
            for worker, normalizer in zip(envs, snapshot["normalizers"]):
                worker._worst_delay_sq = normalizer
            with open(log_path, newline="") as f:
                old_rows = list(csv.reader(f))
            if old_rows[0] != header or len(old_rows) < ep + 1:
                raise ValueError("Training log does not match the checkpoint")
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerows(old_rows[:ep + 1])
            episodes_done = snapshot.get("episodes_done")
            if episodes_done is None:  # snapshot from before the counter existed
                episodes_done = (sum(row[-1] == "1" for row in old_rows[1:ep + 1])
                                 if args.workers > 1 else ep)
            print(f"[Train] Resumed episode {ep}, decision {agent.decisions}")
            if pending_validations:
                # A checkpoint can be written while frozen policies are being
                # scored. Drop newer scores, then replay those validations.
                if os.path.exists(validation_log_path):
                    first_pending = min(job["step"] for job in pending_validations)
                    with open(validation_log_path, newline="") as f:
                        previous = [row for row in csv.reader(f)
                                    if int(row[1]) < first_pending]
                    tmp = validation_log_path + ".tmp"
                    with open(tmp, "w", newline="") as f:
                        csv.writer(f).writerows(previous)
                    os.replace(tmp, validation_log_path)
                for job in pending_validations:
                    submit_validation(dict(job))
        else:
            if os.path.exists(log_path):
                raise FileExistsError("Training output already exists; use --resume-dir")
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

        def save_training_snapshot():
            manifest.flush()  # the demand files this checkpoint has generated
            snapshot = {"config": config, "episode": ep,
                        "episodes_done": episodes_done,
                        "next_seed": next_seed,
                        "best_validation_queue": best_validation_queue,
                        "normalizers": [worker._worst_delay_sq for worker in envs],
                        "agent": agent.training_state()}
            if validation_queue:
                snapshot["pending_validations"] = [
                    {key: value for key, value in job.items() if key != "futures"}
                    for job in validation_queue]
                if math.isfinite(best_validation_queue):
                    snapshot["best_before_validation"] = Path(outdir, "best_weights.pt").read_bytes()
            if process_executors:
                snapshot.update(worker_states=worker_states,
                                worker_seeds=worker_seeds,
                                worker_sim_payloads=worker_sim_payloads)
            temporary = os.path.join(outdir, "training_state.pt.tmp")
            torch.save(snapshot, temporary)
            os.replace(temporary, os.path.join(outdir, "training_state.pt"))
        train_started = time.monotonic()
        deadline = (train_started + args.max_train_seconds
                    if args.max_train_seconds is not None else None)
        def time_budget_reached():
            return (args.max_train_seconds is not None and
                    time.monotonic() - train_started >= args.max_train_seconds)

        if process_executors:
            learner_threads = args.learner_threads or max(
                1, (os.cpu_count() or 2) // 2 - args.workers)
            torch.set_num_threads(learner_threads)
            print(f"[Train] rollouts: {args.rollout_mode} | learner threads: {learner_threads} | "
                  f"replay ratio: {args.replay_ratio}")
        # Resumed workers rebuild their simulation from the checkpoint once.
        payload_pending = [worker_states[i] is not None for i in range(args.workers)]

        learner = None  # LearnerThread of the async Python path
        learner_stats = {}

        async def validate_and_checkpoint(completed_episode, finished):
            """Validation and checkpoint cadence of one learned rollout."""
            await collect_validation()
            marker = ""
            if (not time_budget_reached() and
                    ((completed_episode and
                      episodes_done % args.validate_every == 0) or finished)):
                if learner is None:
                    marker = await validate_policy(episodes_done)
                else:
                    with learner.lock:  # a consistent copy of the weights
                        marker = await validate_policy(episodes_done)
            if (completed_episode and episodes_done % args.checkpoint_every == 0
                    and not time_budget_reached() and not finished):
                await save_checkpoint()
            return marker

        async def save_checkpoint():
            """Checkpoint after every handed-over rollout has been learned, so
            the replay matches the worker states saved beside it."""
            if learner is None:
                agent.save(os.path.join(outdir, "checkpoint.pt"))
                save_training_snapshot()
                return
            await learner.wait_below(1)
            with learner.lock:
                agent.save(os.path.join(outdir, "checkpoint.pt"))
                save_training_snapshot()

        async def run_async_rollouts():
            """Each worker process gets its next rollout as soon as its last
            one has been learned; nobody waits for the slowest simulation.

            The learner runs in this process between results while the other
            workers keep simulating. Arrival order depends on timing, so the
            order of replay insertions is not reproducible run to run
            (``--rollout-mode sync`` keeps the lock-step order).
            """
            nonlocal ep, episodes_done, next_seed, learner
            loop = asyncio.get_running_loop()
            inflight = {}
            reserved = [0]  # decisions promised to rollouts in flight
            if args.learner == "thread":
                learner = LearnerThread(agent, args)
                learner.start()
            max_lag = args.max_learner_lag or args.workers
            held_back = [0.0]  # seconds new rollouts waited for the learner

            def actor_payload(i):
                seed = actor_seed(worker_seeds[i], agent.decisions, i)
                if learner is not None and args.algorithm == "rainbow":
                    return ("weights", learner.weights, seed)
                if learner is not None:
                    with learner.lock:
                        actor = make_actor(agent_class, agent, args, seed)
                else:
                    actor = make_actor(agent_class, agent, args, seed)
                return pickle.dumps(actor, protocol=pickle.HIGHEST_PROTOCOL)

            def submit(i):
                nonlocal next_seed
                if (agent.decisions >= args.total_steps or time_budget_reached() or
                        (deadline is not None and deadline - time.monotonic() < 5.0)):
                    return False
                quota = min(args.rollout_decisions,
                            args.total_steps - agent.decisions - reserved[0])
                if quota <= 0:
                    return False
                parameters = None
                if worker_states[i] is None:
                    started = next_seed - args.seed_base - 1
                    if args.episodes and started >= args.episodes:
                        return False
                    worker_seeds[i] = next_seed
                    next_seed += 1
                    parameters = episode_parameters(worker_seeds[i])
                payload = worker_sim_payloads[i] if payload_pending[i] else None
                payload_pending[i] = False
                future = loop.run_in_executor(
                    process_executors[i], collect_python_rollout_process,
                    actor_payload(i),
                    args, worker_seeds[i], quota,
                    args.algorithm != "qlearning" and len(agent.buffer) < args.warmup,
                    parameters, deadline, worker_states[i], payload)
                inflight[future] = (i, quota)
                reserved[0] += quota
                return True

            for i in range(args.workers):
                submit(i)
            try:
                while inflight:
                    done, _ = await asyncio.wait(
                        list(inflight), return_when=asyncio.FIRST_COMPLETED)
                    for future in sorted(done, key=lambda f: inflight[f][0]):
                        i, quota = inflight.pop(future)
                        reserved[0] -= quota
                        item = future.result()
                        if item["decisions"] == 0:
                            continue
                        ep += 1
                        seed = item["seed"]
                        if learner is not None:
                            learner.check()
                            learner.submit(item["transitions"])
                        else:
                            for observation, action, reward, next_observation, seconds, mask, terminal in item["transitions"]:
                                emitted = agent.store(observation, action, reward, next_observation, terminal,
                                                      seconds=seconds, next_mask=mask)
                                learn_emitted(agent, args, emitted)
                            if args.algorithm == "rainbow":
                                learn_emitted(agent, args, agent.finish_episode())
                        for _ in range(item["decisions"]):
                            agent.decay_epsilon()
                        state = item["state"]
                        completed_episode = state["cycle"] >= args.episode_horizon
                        worker_states[i] = None if completed_episode else state
                        if completed_episode:
                            worker_seeds[i] = None
                            episodes_done += 1
                        worker_sim_payloads[i] = (None if completed_episode else
                                                  item["sim_payload"])
                        finished = (agent.decisions >= args.total_steps or
                                    (args.episodes and episodes_done >= args.episodes))
                        row = [ep, seed, state["cycle"], item["decisions"], round(item["reward"], 2),
                               round(agent.eps, 4), round(agent.avg_loss, 5), agent.update_count,
                               state["total_vehicles_passed"], state["total_vehicles_created"],
                               state["average_waiting_time"], state["max_waiting_time"],
                               state["average_speed"], state["average_travel_time"],
                               state["current_queue_length"], round(item["wall"], 1),
                               item["simulated_cycles"], int(completed_episode)]
                        with open(log_path, "a", newline="") as f:
                            csv.writer(f).writerow(row)
                        marker = await validate_and_checkpoint(completed_episode, finished)
                        print(f"[Rollout {ep:5d} | worker {i} | episode {episodes_done:4d} | "
                              f"step {agent.decisions}/{args.total_steps}] "
                              f"reward={item['reward']:8.1f} | decisions={item['decisions']:3d} | "
                              f"eps={agent.eps:.3f} | loss={agent.avg_loss:.4f} | "
                              f"passed={state['total_vehicles_passed']:3d} | "
                              f"queue={state['current_queue_length']:2d} | "
                              f"{item['wall']:5.1f}s{marker}", flush=True)
                        if not finished:
                            if learner is not None:
                                # Keep the replay ratio: hold new work back
                                # while the learner is ``max_lag`` rollouts behind.
                                held_back[0] += await learner.wait_below(max_lag)
                            submit(i)
                if learner is not None:
                    await learner.wait_below(1)
            except BaseException:
                for future in inflight:
                    future.cancel()
                raise
            finally:
                if learner is not None:
                    learner.stop()
                    learner.join(timeout=600)
                    learner_stats.update(
                        learner="thread", learner_busy_seconds=round(learner.busy_seconds, 2),
                        rollouts_held_back_seconds=round(held_back[0], 2), max_learner_lag=max_lag)
                    learner.check()
                    learner = None

        if process_executors and args.rollout_mode == "async":
            await run_async_rollouts()
            learner_stats["rollout_phase_seconds"] = round(time.monotonic() - train_started, 2)
            await collect_validation(wait=True)
            # All in-flight results are now reflected in one resumable checkpoint.
            agent.save(os.path.join(outdir, "checkpoint.pt"))
            save_training_snapshot()

        while (not (process_executors and args.rollout_mode == "async") and
               agent.decisions < args.total_steps and
               (args.episodes == 0 or episodes_done < args.episodes) and
               not time_budget_reached()):
            if args.workers > 1:
                if deadline is not None and deadline - time.monotonic() < 5.0:
                    break
                # ``--episodes`` bounds episodes, not rollouts: a worker whose
                # episode has ended starts a new one only while episodes
                # remain; unfinished episodes always continue.
                remaining = args.total_steps - agent.decisions
                slots = []
                for i in range(args.workers):
                    if len(slots) >= remaining:
                        break
                    if worker_states[i] is None:
                        started = next_seed - args.seed_base - 1
                        if args.episodes and started >= args.episodes:
                            continue
                        worker_seeds[i] = next_seed
                        next_seed += 1
                    slots.append(i)
                if not slots:
                    break
                batch_size = len(slots)
                quotas = [min(args.rollout_decisions,
                              remaining // batch_size + int(k < remaining % batch_size))
                          for k in range(batch_size)]
                actors = [make_actor(agent_class, agent, args,
                                     actor_seed(worker_seeds[i], agent.decisions, i))
                          for i in slots]
                # All actors see the same frozen learner; independent GAMA
                # processes collect in parallel. Learn in seed order below.
                if process_executors:
                    loop = asyncio.get_running_loop()
                    tasks = [loop.run_in_executor(
                        process_executors[i], collect_python_rollout_process,
                        pickle.dumps(actors[k], protocol=pickle.HIGHEST_PROTOCOL),
                        args, worker_seeds[i], quotas[k],
                        args.algorithm != "qlearning" and len(agent.buffer) < args.warmup,
                        episode_parameters(worker_seeds[i]) if worker_states[i] is None else None,
                        deadline, worker_states[i], worker_sim_payloads[i])
                             for k, i in enumerate(slots)]
                else:
                    tasks = [asyncio.create_task(collect_rollout(
                        envs[i], actors[k], args, worker_seeds[i],
                        quotas[k], args.algorithm != "qlearning" and
                        len(agent.buffer) < args.warmup,
                        episode_parameters(worker_seeds[i]) if worker_states[i] is None else None,
                        deadline=deadline, initial_state=worker_states[i]))
                             for k, i in enumerate(slots)]
                try:
                    collected = await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                for i, item in zip(slots, collected):
                    if item["decisions"] == 0:
                        continue
                    ep += 1
                    seed = item["seed"]
                    for observation, action, reward, next_observation, seconds, mask, terminal in item["transitions"]:
                        emitted = agent.store(observation, action, reward, next_observation, terminal,
                                              seconds=seconds, next_mask=mask)
                        learn_emitted(agent, args, emitted)
                    if args.algorithm == "rainbow":
                        learn_emitted(agent, args, agent.finish_episode())
                    for _ in range(item["decisions"]):
                        agent.decay_epsilon()
                    state = item["state"]
                    completed_episode = state["cycle"] >= args.episode_horizon
                    worker_states[i] = None if completed_episode else state
                    if completed_episode:
                        worker_seeds[i] = None
                        episodes_done += 1
                    finished = (agent.decisions >= args.total_steps or
                                (args.episodes and episodes_done >= args.episodes))
                    if process_executors:
                        worker_sim_payloads[i] = (None if completed_episode else
                                                  item["sim_payload"])
                    row = [ep, seed, state["cycle"], item["decisions"], round(item["reward"], 2),
                           round(agent.eps, 4), round(agent.avg_loss, 5), agent.update_count,
                           state["total_vehicles_passed"], state["total_vehicles_created"],
                           state["average_waiting_time"], state["max_waiting_time"],
                           state["average_speed"], state["average_travel_time"],
                           state["current_queue_length"], round(item["wall"], 1),
                           item["simulated_cycles"], int(completed_episode)]
                    with open(log_path, "a", newline="") as f:
                        csv.writer(f).writerow(row)
                    await collect_validation()
                    marker = ""
                    if (not time_budget_reached() and
                            ((completed_episode and
                              episodes_done % args.validate_every == 0) or finished)):
                        marker = await validate_policy(
                            episodes_done, reset_worker0=not process_executors)
                    if (time_budget_reached() or finished or
                            (completed_episode and
                             episodes_done % args.checkpoint_every == 0)):
                        agent.save(os.path.join(outdir, "checkpoint.pt"))
                        save_training_snapshot()
                    print(f"[Rollout {ep:5d} | episode {episodes_done:4d} | step {agent.decisions}/{args.total_steps}] "
                          f"reward={item['reward']:8.1f} | decisions={item['decisions']:3d} | "
                          f"eps={agent.eps:.3f} | loss={agent.avg_loss:.4f} | "
                          f"passed={state['total_vehicles_passed']:3d} | "
                          f"queue={state['current_queue_length']:2d} | "
                          f"{item['wall']:5.1f}s{marker}")
                continue
            ep += 1
            t0 = time.time()
            seed = args.seed_base + ep
            state = await env.reset(seed=seed, parameters=episode_parameters(seed))
            # The controller first requests an action at cycle 100.  A call to
            # env.step() before then would store a transition for an action
            # that GAMA has not applied.
            await env.step_cycles(state["next_decision_in_cycles"])
            state = await env.read_state()
            if not state["dqn_waiting_action"]:
                raise RuntimeError("GAMA did not reach the first DQN decision")

            ep_reward = 0.0
            decisions = 0
            done = False
            while not done and agent.decisions < args.total_steps:
                state_vec = featurize(state, getattr(env, "queue_feature_scale", "linear"))
                # The paper fills N_min with uniformly random legal transitions
                # before starting gradient updates, independently of epsilon.
                action = agent.act(state_vec, action_mask=state["action_mask"],
                                   force_random=args.algorithm != "qlearning" and
                                   len(agent.buffer) < args.warmup)
                next_state, reward, _, info = await env.step(
                    action, horizon=args.episode_horizon, current_state=state)
                if info["bootstrap_valid"]:
                    emitted = agent.store(state_vec, action, reward, featurize(next_state, getattr(env, "queue_feature_scale", "linear")),
                                          not info["bootstrap_valid"],
                                          seconds=info["elapsed_seconds"],
                                          next_mask=next_state["action_mask"])
                    learn_emitted(agent, args, emitted)
                agent.decay_epsilon()
                if agent.decisions in (1_000_000, 5_000_000):
                    agent.save(os.path.join(outdir, f"{args.algorithm}_{agent.decisions}_steps.pt"))
                ep_reward += reward
                decisions += 1
                state = next_state
                done = state["cycle"] >= args.episode_horizon or time_budget_reached()

            if args.algorithm == "rainbow":
                learn_emitted(agent, args, agent.finish_episode())

            wall = time.time() - t0
            row = [
                ep, seed, state["cycle"], decisions, round(ep_reward, 2), round(agent.eps, 4),
                round(agent.avg_loss, 5), agent.update_count,
                state["total_vehicles_passed"], state["total_vehicles_created"],
                state["average_waiting_time"], state["max_waiting_time"],
                state["average_speed"], state["average_travel_time"],
                state["current_queue_length"], round(wall, 1),
                state["cycle"], int(state["cycle"] >= args.episode_horizon),
            ]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            await collect_validation()
            marker = ""
            if (not time_budget_reached() and
                    (ep % args.validate_every == 0 or agent.decisions >= args.total_steps or
                     ep == args.episodes)):
                marker = await validate_policy(ep)
            if (time_budget_reached() or ep % args.checkpoint_every == 0 or
                    agent.decisions >= args.total_steps or ep == args.episodes):
                agent.save(os.path.join(outdir, "checkpoint.pt"))
                save_training_snapshot()

            wait_display = (f"{state['average_waiting_time']:.2f}s"
                            if math.isfinite(state["average_waiting_time"]) else "n/a")
            print(f"[Ep {ep:5d} | step {agent.decisions}/{args.total_steps}] reward={ep_reward:8.1f} | "
                  f"decisions={decisions:3d} | eps={agent.eps:.3f} | "
                  f"loss={agent.avg_loss:.4f} | passed={state['total_vehicles_passed']:3d} | "
                  f"wait={wait_display} | "
                  f"queue={state['current_queue_length']:2d} | {wall:5.1f}s{marker}")

            episodes_done = ep
        if validation_queue:
            # The loops above never waited for validation; collect what is still running.
            await collect_validation(wait=True)
            save_training_snapshot()
        manifest.flush()
        # 4. Luu weights cuoi cung
        final_path = os.path.join(outdir, "dqn_weights.pt")
        agent.save(final_path)
        with open(log_path, newline="") as stream:
            completed_rows = list(csv.DictReader(stream))
        train_elapsed = time.monotonic() - train_started
        dt = getattr(env, "step_seconds", .2 if args.scenario == "lvl" else .1)
        simulated_seconds = sum(int(row["simulated_cycles"]) for row in completed_rows) * dt
        summary = {"algorithm": args.algorithm, "scenario": args.scenario,
                   "backend": args.backend,
                   "python_sim_mode": args.python_sim_mode,
                   "fidelity": ("LVL IDM/MOBIL paired approximation" if args.scenario == "lvl"
                                else env.fidelity if args.backend == "python"
                                else "GAMA 2025.6.4+"),
                   "reward_mode": args.reward_mode,
                   "workers": args.workers,
                   "episodes": sum(int(row["completed_episode"]) for row in completed_rows),
                   "rollouts": len(completed_rows),
                   "decisions": agent.decisions, "updates": agent.update_count,
                   "train_wall_seconds": round(train_elapsed, 2),
                   "aggregate_simulated_seconds": round(simulated_seconds, 2),
                   "decisions_per_second": round(agent.decisions / train_elapsed, 5),
                   # Rollout phase only: without the final wait for queued
                   # validations and the last checkpoint.
                   "rollout_phase_seconds": learner_stats.get("rollout_phase_seconds"),
                   "decisions_per_second_rollout_phase": (
                       round(agent.decisions / learner_stats["rollout_phase_seconds"], 5)
                       if learner_stats.get("rollout_phase_seconds") else None),
                   **{key: value for key, value in learner_stats.items()
                      if key != "rollout_phase_seconds"},
                   "simulated_seconds_per_wall_second": round(simulated_seconds / train_elapsed, 5),
                   "training_config": config,
                   "source_sha256": config["code_sha256"],
                   "runtime": {"python": platform.python_version(),
                               "numpy": np.__version__, "torch": torch.__version__},
                   "demand_manifest": demand_manifest,
                   "training_seed_range": (
                       [min(int(row["seed"]) for row in completed_rows),
                        max(int(row["seed"]) for row in completed_rows)]
                       if completed_rows else []),
                   "validation_seeds": list(validation_seeds),
                   "validation_levels": [list(level) for level in args.validation_levels],
                   "time_metric_semantics": (
                       "LVL average_waiting_time is cumulative queue-time integral / created; "
                       "average_travel_time is cumulative system-time integral / passed; "
                       "neither is a completed-vehicle mean; max_waiting_time is unavailable")
                       if args.scenario == "lvl" else "GAMA/backend-defined state metrics"}
        with open(os.path.join(outdir, "training_summary.json"), "w") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
        print("\n[Train] HOAN TAT.")
        print(f"[Train] Weights cuoi cung : {final_path}")
        best_path = os.path.join(outdir, "best_weights.pt")
        if os.path.exists(best_path):
            print(f"[Train] Weights tot nhat  : {best_path}")
        else:
            print("[Train] Chua co best_weights.pt (validation chua chay)")
        print(f"[Train] Log huan luyen    : {log_path}")
        print(f"[Train] Train wall time    : {train_elapsed:.1f}s")
    finally:
        for executor in process_executors + validation_executors:
            executor.shutdown(wait=True, cancel_futures=True)
        if args.start_server and not args.keep_server:
            # These workspaces and JVMs belong only to this training run.
            # GAMA's stop/disconnect handlers can dispose an agent scope while
            # a driving reflex is still active, producing spurious Java NPEs.
            # Terminate the temporary JVMs before dropping their WebSockets.
            stop_results = await asyncio.gather(
                *(asyncio.to_thread(manager.stop, force=True) for manager in managers),
                return_exceptions=True)
            for env in envs:
                env.disconnect()
            for result in stop_results:
                if isinstance(result, BaseException):
                    raise result
        else:
            await asyncio.gather(*(env.close() for env in envs))


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
