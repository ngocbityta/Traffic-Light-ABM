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
import json
import math
import os
import pickle
import time
from pathlib import Path

import torch

from three_dqn_agent import ThreeDQNAgent, FlatDQNAgent
from rainbow_agent import RainbowAgent
from hybrid_agent import SpatialHybridAgent
from tabular_agent import TabularQAgent
from gama_env import GamaTrafficEnv, featurize
from python_env import PythonTrafficEnv
from lvl_env import LVLGamaEnv, LVLPythonEnv
from server_manager import GamaServerManager
from evaluate import evaluate_episode
from demand import generate_demand, demand_parameters
from LeVanLuongIntersection.demand import generate_demand as generate_lvl_demand
from LeVanLuongIntersection.demand import demand_parameters as lvl_demand_parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
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
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--start-server", action="store_true",
                   help="Tu khoi dong gama-server (neu chua chay)")
    p.add_argument("--workers", type=int, default=1,
                   help="Independent simulation workers; one learner merges batches in seed order")
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
                   help="Reserved held-out seed range; validation uses base+1 and base+2")
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
    p.add_argument("--eps-decay-steps", type=float, default=2_000_000)
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
    args = p.parse_args()
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
            min(args.r1_ref, args.r2_ref, args.b_ref) <= 0 or
            min(args.r2_lambda, args.r3_beta, args.pressure_lambda) < 0):
        p.error("Horizon must exceed 100 cycles; total-steps and validate-every must be positive")
    if args.backend == "python" and args.start_server:
        p.error("--start-server is only valid with --backend gama")
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
    possible_episodes = min(args.total_steps, args.episodes) if args.episodes else args.total_steps
    training_end = args.seed_base + possible_episodes
    if any(args.seed_base < args.validation_seed_base + offset <= training_end
           for offset in (1, 2)):
        p.error("Validation seeds overlap the possible training seed range")
    if args.gamma is None:
        args.gamma = .97 if args.reward_mode in ("queue_time", "total_wait", "r1", "r2", "r3", "r4") else .99
    if args.lr is None:
        args.lr = 0.1 if args.algorithm == "qlearning" else 1e-4
    return args


def training_config(args):
    fields = ("algorithm", "scenario", "backend", "episode_horizon", "max_train_seconds", "validate_every", "checkpoint_every",
              "workers", "rollout_decisions", "gama_heap_mb", "fast_simulation", "seed_base",
              "validation_seed_base", "lr", "gamma",
              "reward_mode", "r1_ref", "r2_ref", "b_ref", "r2_lambda", "r3_beta",
              "pressure_lambda", "demand_mode", "max_delay_sq", "buffer_size", "warmup", "batch_size",
              "tau", "eps_start", "eps_decay_steps", "eps_min", "model_path")
    config = {name: getattr(args, name) for name in fields}
    if args.algorithm == "rainbow":
        config.update({name: getattr(args, name) for name in
                       ("n_step", "per_alpha", "per_beta_start", "per_beta_steps",
                        "c51_atoms", "c51_v_min", "c51_v_max", "target_update")})
    files = ("Traffic/includes/collision_avoidance.gaml",
             "Traffic/models/intersection-dqn-python.gaml",
             "dqn_py/gama_env.py", "dqn_py/python_env.py", "dqn_py/evaluate.py",
             "dqn_py/server_manager.py", "dqn_py/demand.py",
             "dqn_py/three_dqn_agent.py",
             "dqn_py/tabular_agent.py", "dqn_py/rainbow_agent.py", "dqn_py/train.py")
    files = files + ("dqn_py/hybrid_agent.py",)
    if args.scenario == "lvl":
        files += ("dqn_py/lvl_env.py", "LeVanLuongIntersection/python_sim.py",
                  "LeVanLuongIntersection/demand.py",
                  "LeVanLuongIntersection/models/LeVanLuong.gaml",
                  "LeVanLuongIntersection/includes/roads.geojson")
    config["code_sha256"] = {name: hashlib.sha256(
        (Path(BASE_DIR).parent / name).read_bytes()).hexdigest() for name in files}
    return config


def make_actor(agent_class, learner, args, seed):
    """Freeze a small CPU policy for one rollout; never copy the learner's replay."""
    actor = agent_class(lr=args.lr, gamma=args.gamma, buffer_size=1,
                        warmup=1, batch_size=1, tau=args.tau,
                        eps_start=args.eps_start, eps_decay_steps=args.eps_decay_steps,
                        eps_min=args.eps_min, seed=seed, reward_mode=args.reward_mode,
                        **({} if args.algorithm == "qlearning" else {"device": torch.device("cpu")}),
                        **spatial_kwargs(args))
    actor.eps = learner.eps
    actor.decisions = learner.decisions
    if args.algorithm == "qlearning":
        actor.table = {key: values.copy() for key, values in learner.table.items()}
    else:
        actor.online_net.load_state_dict(learner.online_net.state_dict())
        if args.algorithm != "rainbow":
            actor.online_net.eval()
    return actor


def reward_settings(args):
    return dict(r1_ref=args.r1_ref, r2_ref=args.r2_ref, b_ref=args.b_ref,
                r2_lambda=args.r2_lambda, r3_beta=args.r3_beta)


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


def learn_emitted(agent, args, emitted):
    updates = emitted if args.algorithm == "rainbow" else 1
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
        observation = featurize(state)
        action = actor.act(observation, action_mask=state["action_mask"],
                           force_random=force_warmup)
        next_state, reward, _, info = await env.step(
            action, horizon=args.episode_horizon, current_state=state)
        if info["bootstrap_valid"]:
            transitions.append((observation, action, reward, featurize(next_state),
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


def collect_python_rollout_process(actor_payload, args, seed, decision_limit,
                                   force_warmup, parameters, deadline, initial_state,
                                   sim_payload):
    """Keep one Python simulation alive in each dedicated worker process."""
    global _PYTHON_PROCESS_ENV
    if _PYTHON_PROCESS_ENV is None:
        _PYTHON_PROCESS_ENV = LVLPythonEnv(
            model_path=args.model_path, reward_mode=args.reward_mode,
            max_delay_sq=args.max_delay_sq,
            state_mode="spatial" if args.algorithm in ("rainbow", "hybrid") else "dtse",
            reset_mode="fresh", fast_training=args.fast_simulation,
            **reward_settings(args))
    if initial_state is not None and _PYTHON_PROCESS_ENV.sim is None:
        if sim_payload is None:
            raise RuntimeError("Missing Python simulation state for a resumed rollout")
        _PYTHON_PROCESS_ENV.sim = pickle.loads(sim_payload)
    actor = pickle.loads(actor_payload)
    result = asyncio.run(collect_rollout(
        _PYTHON_PROCESS_ENV, actor, args, seed, decision_limit, force_warmup,
        parameters=parameters, deadline=deadline, initial_state=initial_state))
    result["sim_payload"] = pickle.dumps(_PYTHON_PROCESS_ENV.sim)
    return result

async def amain(args):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.resume_dir or args.output_dir or os.path.join(
        BASE_DIR, "run", f"{args.algorithm}_{stamp}")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    print(f"[Train] Ket qua se luu tai: {outdir}")
    if args.backend == "python":
        print("[Train] Python/GAMA parity is measured separately; vehicle trajectories may differ")
    config = training_config(args)
    def episode_parameters(seed):
        if args.demand_mode == "model":
            return None
        path = Path(outdir) / "demand" / f"seed_{seed}_h{args.episode_horizon}.csv"
        if args.scenario == "lvl":
            generate_lvl_demand(path, seed, args.episode_horizon)
            return lvl_demand_parameters(path)
        generate_demand(path, seed, args.episode_horizon)
        return demand_parameters(path)

    # 1. Server
    managers = []
    envs = []
    process_executors = []
    try:
        if args.backend == "python" and args.workers > 1:
            process_executors = [concurrent.futures.ProcessPoolExecutor(max_workers=1)
                                 for _ in range(args.workers)]
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
                                 **reward_settings(args))
            await env.setup()
            envs.append(env)
        env = envs[0]
        agent_class = {"qlearning": TabularQAgent, "dqn": FlatDQNAgent,
                       "3dqn": ThreeDQNAgent, "rainbow": RainbowAgent,
                       "hybrid": SpatialHybridAgent}[args.algorithm]
        agent = agent_class(
            lr=args.lr, gamma=args.gamma, buffer_size=args.buffer_size,
            warmup=args.warmup, batch_size=args.batch_size, tau=args.tau,
            eps_start=args.eps_start, eps_decay_steps=args.eps_decay_steps,
            eps_min=args.eps_min, seed=args.seed_base, reward_mode=args.reward_mode,
            **spatial_kwargs(args))
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
        ep = 0
        next_seed = args.seed_base + 1
        worker_states = [None] * args.workers
        worker_seeds = [None] * args.workers
        worker_sim_payloads = [None] * args.workers
        if args.resume_dir:
            snapshot = torch.load(os.path.join(outdir, "training_state.pt"),
                                  map_location=agent.device, weights_only=False)
            if snapshot["config"] != config:
                raise ValueError("Resume requires identical training settings and source code")
            agent.restore_training(snapshot["agent"])
            ep = snapshot["episode"]
            next_seed = snapshot["next_seed"]
            best_validation_queue = snapshot["best_validation_queue"]
            if process_executors:
                worker_states = snapshot["worker_states"]
                worker_seeds = snapshot["worker_seeds"]
                worker_sim_payloads = snapshot["worker_sim_payloads"]
            for worker, normalizer in zip(envs, snapshot["normalizers"]):
                worker._worst_delay_sq = normalizer
            with open(log_path, newline="") as f:
                old_rows = list(csv.reader(f))
            if old_rows[0] != header or len(old_rows) < ep + 1:
                raise ValueError("Training log does not match the checkpoint")
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerows(old_rows[:ep + 1])
            print(f"[Train] Resumed episode {ep}, decision {agent.decisions}")
        else:
            if os.path.exists(log_path):
                raise FileExistsError("Training output already exists; use --resume-dir")
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

        def save_training_snapshot():
            snapshot = {"config": config, "episode": ep,
                        "next_seed": next_seed,
                        "best_validation_queue": best_validation_queue,
                        "normalizers": [worker._worst_delay_sq for worker in envs],
                        "agent": agent.training_state()}
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

        while (agent.decisions < args.total_steps and
               (args.episodes == 0 or ep < args.episodes) and
               not time_budget_reached()):
            if args.workers > 1:
                if deadline is not None and deadline - time.monotonic() < 5.0:
                    break
                available_episodes = (args.episodes - ep) if args.episodes else args.workers
                batch_size = min(args.workers, available_episodes,
                                 args.validate_every - ep % args.validate_every,
                                 args.checkpoint_every - ep % args.checkpoint_every,
                                 args.total_steps - agent.decisions)
                remaining = args.total_steps - agent.decisions
                quotas = [min(args.rollout_decisions,
                              remaining // batch_size + int(i < remaining % batch_size))
                          for i in range(batch_size)]
                for i in range(batch_size):
                    if worker_states[i] is None:
                        worker_seeds[i] = next_seed
                        next_seed += 1
                actors = [make_actor(agent_class, agent, args, worker_seeds[i])
                          for i in range(batch_size)]
                # All actors see the same frozen learner; independent GAMA
                # processes collect in parallel. Learn in seed order below.
                if process_executors:
                    loop = asyncio.get_running_loop()
                    tasks = [loop.run_in_executor(
                        process_executors[i], collect_python_rollout_process,
                        pickle.dumps(actors[i]), args, worker_seeds[i], quotas[i],
                        args.algorithm != "qlearning" and len(agent.buffer) < args.warmup,
                        episode_parameters(worker_seeds[i]) if worker_states[i] is None else None,
                        deadline, worker_states[i], worker_sim_payloads[i])
                             for i in range(batch_size)]
                else:
                    tasks = [asyncio.create_task(collect_rollout(
                        envs[i], actors[i], args, worker_seeds[i],
                        quotas[i], args.algorithm != "qlearning" and
                        len(agent.buffer) < args.warmup,
                        episode_parameters(worker_seeds[i]) if worker_states[i] is None else None,
                        deadline=deadline, initial_state=worker_states[i]))
                             for i in range(batch_size)]
                try:
                    collected = await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                for i, item in enumerate(collected):
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
                    marker = ""
                    if (not time_budget_reached() and
                            (ep % args.validate_every == 0 or agent.decisions >= args.total_steps or
                             ep == args.episodes)):
                        normalizer = env._worst_delay_sq
                        scores = []
                        try:
                            for validation_seed in (args.validation_seed_base + 1,
                                                    args.validation_seed_base + 2):
                                last, _, _, _ = await evaluate_episode(
                                    env, agent, validation_seed, args.episode_horizon,
                                    DiscardSamples(), 0, collect_samples=False,
                                    parameters=episode_parameters(validation_seed))
                                wait = (env.reward_cost_from_state(last)
                                        if args.reward_mode in ("r1", "r2", "r3", "r4")
                                        else last["queue_time_integral"])
                                if args.reward_mode == "total_wait":
                                    wait += last["external_wait_integral"]
                                dt = getattr(env, "step_seconds", .2 if args.scenario == "lvl" else .1)
                                scores.append(wait / (args.episode_horizon * dt))
                        finally:
                            env._worst_delay_sq = normalizer
                            if not process_executors:
                                worker_states[0] = None
                                worker_seeds[0] = None
                        score = sum(scores) / len(scores)
                        with open(os.path.join(outdir, "validation_log.csv"), "a", newline="") as f:
                            csv.writer(f).writerow([ep, agent.decisions, *scores, score])
                        if score < best_validation_queue:
                            best_validation_queue = score
                            agent.save(os.path.join(outdir, "best_weights.pt"))
                            marker = "  <- best validation queue"
                    if (time_budget_reached() or ep % args.checkpoint_every == 0 or
                            agent.decisions >= args.total_steps or ep == args.episodes):
                        agent.save(os.path.join(outdir, "checkpoint.pt"))
                        save_training_snapshot()
                    print(f"[Rollout {ep:5d} | step {agent.decisions}/{args.total_steps}] "
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
                state_vec = featurize(state)
                # The paper fills N_min with uniformly random legal transitions
                # before starting gradient updates, independently of epsilon.
                action = agent.act(state_vec, action_mask=state["action_mask"],
                                   force_random=args.algorithm != "qlearning" and
                                   len(agent.buffer) < args.warmup)
                next_state, reward, _, info = await env.step(
                    action, horizon=args.episode_horizon, current_state=state)
                if info["bootstrap_valid"]:
                    emitted = agent.store(state_vec, action, reward, featurize(next_state),
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

            marker = ""
            if (not time_budget_reached() and
                    (ep % args.validate_every == 0 or agent.decisions >= args.total_steps or
                     ep == args.episodes)):
                # Validation must not change the training reward normalizer.
                normalizer = env._worst_delay_sq
                scores = []
                try:
                    for validation_seed in (args.validation_seed_base + 1,
                                            args.validation_seed_base + 2):
                        last, _, _, _ = await evaluate_episode(env, agent, validation_seed,
                            args.episode_horizon, DiscardSamples(), 0,
                            collect_samples=False,
                            parameters=episode_parameters(validation_seed))
                        wait = (env.reward_cost_from_state(last)
                                if args.reward_mode in ("r1", "r2", "r3", "r4")
                                else last["queue_time_integral"])
                        if args.reward_mode == "total_wait":
                            wait += last["external_wait_integral"]
                        dt = getattr(env, "step_seconds", .2 if args.scenario == "lvl" else .1)
                        scores.append(wait / (args.episode_horizon * dt))
                finally:
                    env._worst_delay_sq = normalizer
                score = sum(scores) / len(scores)
                with open(os.path.join(outdir, "validation_log.csv"), "a", newline="") as f:
                    csv.writer(f).writerow([ep, agent.decisions, *scores, score])
                if score < best_validation_queue:
                    best_validation_queue = score
                    agent.save(os.path.join(outdir, "best_weights.pt"))
                    marker = "  <- best validation queue"
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
                   "simulated_seconds_per_wall_second": round(simulated_seconds / train_elapsed, 5)}
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
        for executor in process_executors:
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
