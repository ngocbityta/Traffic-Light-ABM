#!/usr/bin/env python3
"""Train and compare four policies with paired demand, timing and held-out seeds.

All policies use intersection-benchmark.gaml through the same action interface.
Tabular Q-learning and DQN learn off-policy from the same training transitions.
Evaluation never updates either policy. Demand CSVs include rejected arrivals,
which stay in external FIFO queues. GAMA CSV format reference:
https://gama-platform.org/wiki/FileTypes
"""
import argparse
import asyncio
import csv
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train" / "logic"))
from train.logic.dqn_agent import QNetwork
from train.logic.gama_env import GamaTrafficEnv, STATE_FIELDS, STATE_PACK_EXPR
from train.logic.server_manager import GamaServerManager
from train.logic.demand import generate_demand

METHODS = ("Fixed-Time", "MaxPressure", "Q-Learning", "Deep Q-Learning")
EXTRA_FIELDS = [
    "through_queue_ns", "through_queue_ew", "through_approach_ns", "through_approach_ew",
    "left_queue_ns", "left_queue_ew", "downstream_ns", "downstream_ew",
    "queue_time_integral", "external_wait_integral", "cumulative_max_wait",
    "ground_requests", "ground_pending", "bypass_requests", "bypass_pending",
]
PACK = STATE_PACK_EXPR + ''.join(f' + "|" + {name}' for name in EXTRA_FIELDS)
MODEL = ROOT / "train/benchmarks/gama/models/intersection-benchmark.gaml"
SOURCE_PATHS = (MODEL, ROOT / "train/benchmarks/gama/models/intersection-dqn-legacy.gaml",
                ROOT / "train/benchmarks/gama/includes/collision_avoidance.gaml",
                ROOT / "train/benchmarks/gama/includes/doan_scenery.gaml", ROOT / "train/logic/gama_env.py",
                ROOT / "train/logic/dqn_agent.py", ROOT / "train/logic/demand.py", Path(__file__))


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE_PATHS}


class BenchmarkEnv(GamaTrafficEnv):
    def __init__(self, port, manager=None, output=None):
        super().__init__(port=port, model_path=str(MODEL), experiment="benchmark_server", verbose=False)
        self.action_count = 2
        self.manager = manager
        self.output = output
        self.episode_number = 0

    async def read_state(self):
        parts = str(await self._exec_expression(PACK)).split("|")
        fields = STATE_FIELDS + EXTRA_FIELDS
        if len(parts) != len(fields):
            raise RuntimeError(f"Expected {len(fields)} state fields, got {len(parts)}")
        state = {name: (value.strip().lower() == "true" if name == "dqn_waiting_action"
                        else float(value)) for name, value in zip(fields, parts)}
        for name in ("cycle", "global_phase", "green_cycles", "next_decision_in_cycles"):
            state[name] = int(state[name])
        if not all(math.isfinite(v) for v in state.values()):
            raise RuntimeError("Non-finite simulation state")
        return state

    async def reset_demand(self, seed, demand_path):
        for relative, expected in getattr(self, "expected_sources", {}).items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Benchmark source changed during the run: {relative}")
        # GAMA can leave an old simulation worker alive during reload().
        # A fresh owned JVM per episode prevents scope-disposal races.
        if self.exp_id is not None and self.manager is not None:
            await self.close()
            await asyncio.to_thread(self.manager.stop)
            self.manager.log_file = str(self.output / "server_logs" / f"port{self.port}_episode_{self.episode_number + 1:03d}.log")
            Path(self.manager.log_file).parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self.manager.start, reuse_running=False)
            await self.setup()
        self.episode_number += 1
        self._server_messages.clear()
        state = await self.reset(seed=seed, parameters=[
            dict(name="demand_file_path", type="string", value=str(demand_path.resolve()))])
        actual = float(await self._exec_expression("world.seed"))
        if actual != seed:
            raise RuntimeError(f"Simulation seed {actual} != requested {seed}")
        return state


def features(s):
    counts = [s[name] for name in EXTRA_FIELDS[:8]]
    return np.asarray([min(x / 20.0, 3.0) for x in counts] +
                      [s["green_cycles"] / 600.0, float(s["global_phase"] == 0),
                       float(s["global_phase"] == 4)], dtype=np.float32)


def tabular_key(s):
    """Canonical own/opposing axes, with coarse queues, age and exit occupancy."""
    own, other = ("ns", "ew") if s["global_phase"] == 0 else ("ew", "ns")
    level = lambda x: 0 if x < 1 else (1 if x < 5 else 2)
    return (level(s[f"through_queue_{own}"]), level(s[f"through_queue_{other}"]),
            int(s[f"left_queue_{own}"] > 0), int(s[f"left_queue_{other}"] > 0),
            int(s[f"downstream_{own}"] >= 5),
            0 if s["green_cycles"] < 250 else (1 if s["green_cycles"] < 450 else 2))


class Learners:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        self.online = QNetwork(11, (64, 64))
        self.target = QNetwork(11, (64, 64))
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=3e-4)
        self.table = {}
        self.replay = deque(maxlen=20000)
        self.updates = 0
        self.transitions = 0

    def act(self, method, s, epsilon=0.0):
        if epsilon > 0 and method in METHODS[2:] and self.rng.random() < epsilon:
            return self.rng.randrange(2)
        if method == "Fixed-Time":
            return int(s["green_cycles"] >= 300)
        if method == "MaxPressure":
            own, other = ("ns", "ew") if s["global_phase"] == 0 else ("ew", "ns")
            incoming = lambda axis: s[f"through_queue_{axis}"] + s[f"through_approach_{axis}"]
            pressure = lambda axis: incoming(axis) - s[f"downstream_{axis}"]
            return int(pressure(other) > pressure(own) + 2 or
                       (incoming(own) == 0 and (s[f"left_queue_{own}"] > 0 or
                                               incoming(other) > 0 or s[f"left_queue_{other}"] > 0)))
        if method == "Q-Learning":
            return int(np.argmax(self.table.get(tabular_key(s), [0.0, 0.0])))
        with torch.no_grad():
            return int(self.online(torch.from_numpy(features(s))).argmax())

    def learn(self, s, action, reward, next_s, done, seconds):
        # Semi-Markov discount and integrated reward account for unequal durations.
        if action not in (0, 1) or not math.isfinite(reward) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid Q/DQN transition")
        if not done and (next_s["global_phase"] not in (0, 4) or next_s["green_cycles"] >= 600):
            raise ValueError("Q/DQN bootstrap requires a legal decision state")
        discount = 0.97 ** (seconds / 5.0)
        item = (features(s), tabular_key(s), action, reward, features(next_s),
                tabular_key(next_s), done, discount)
        self.replay.append(item)
        self.transitions += 1
        for transition in [item] + self.rng.choices(list(self.replay), k=31):
            _, key, a, r, _, key2, terminal, gamma = transition
            values = self.table.setdefault(key, [0.0, 0.0])
            target = r + (0.0 if terminal else gamma * max(self.table.get(key2, [0.0, 0.0])))
            values[a] += .1 * (target - values[a])
        if len(self.replay) < 64:
            return
        for _ in range(4):
            batch = self.rng.sample(list(self.replay), 64)
            x = torch.from_numpy(np.stack([r[0] for r in batch]))
            a = torch.tensor([r[2] for r in batch])
            reward_t = torch.tensor([r[3] for r in batch], dtype=torch.float32)
            x2 = torch.from_numpy(np.stack([r[4] for r in batch]))
            bootstrap = torch.tensor([0.0 if r[6] else r[7] for r in batch], dtype=torch.float32)
            with torch.no_grad():
                target = reward_t + bootstrap * self.target(x2).max(1).values
            loss = nn.functional.smooth_l1_loss(self.online(x).gather(1, a[:, None]).squeeze(1), target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
            self.optimizer.step()
            self.updates += 1
            if self.updates % 100 == 0:
                self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        torch.save(dict(online=self.online.state_dict(), target=self.target.state_dict(),
                        optimizer=self.optimizer.state_dict(), table=self.table,
                        transitions=self.transitions, updates=self.updates,
                        replay=list(self.replay), rng=self.rng.getstate(),
                        torch_rng=torch.get_rng_state()), path)

    def load_training(self, path):
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.table = checkpoint["table"]
        self.replay.extend(checkpoint["replay"])
        self.rng.setstate(checkpoint["rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        self.transitions, self.updates = checkpoint["transitions"], checkpoint["updates"]

    def policy_digest(self):
        digest = hashlib.sha256()
        for value in self.online.state_dict().values():
            digest.update(value.numpy().tobytes())
        digest.update(repr(sorted(self.table.items())).encode())
        return digest.hexdigest()

    def load_policy(self, path, method):
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        if method == "Q-Learning":
            self.table = checkpoint["table"]
        else:
            self.online.load_state_dict(checkpoint["online"])


async def run_episode(env, learners, method, seed, horizon, demand_path, output,
                      train=False, epsilon=0.0, collect_only=False):
    demand_hash = hashlib.sha256(demand_path.read_bytes()).hexdigest()
    sources = source_hashes()
    if (not train or collect_only) and (output / "metrics.json").exists():
        saved = json.loads((output / "metrics.json").read_text())
        if (saved.get("policy_sha256") == learners.policy_digest() and
                saved.get("demand_sha256") == demand_hash and
                saved.get("source_sha256") == sources and
                saved.get("method") == method and saved.get("seed") == seed and
                saved.get("horizon") == horizon and saved.get("training") == train and
                saved.get("epsilon") == epsilon and
                (not collect_only or (output / "transitions.json").exists()) and
                all((output / name).exists() for name in ("series.csv", "decisions.csv"))):
            print(f"Reuse completed evaluation: {method} seed={seed}", flush=True)
            return saved
    state = await env.reset_demand(seed, demand_path)
    t0 = time.monotonic()
    decisions = []
    samples = []
    pending = None
    total_reward = 0.0
    collected = []
    unseen_states = 0
    before_policy = learners.policy_digest()
    next_progress = 1000
    while state["cycle"] < horizon:
        if state["dqn_waiting_action"]:
            if pending is not None:
                prev, action = pending
                reward = -(state["queue_time_integral"] - prev["queue_time_integral"]) / 50.0
                total_reward += reward
                if train and not collect_only:
                    learners.learn(prev, action, reward, state, False,
                                   (state["cycle"] - prev["cycle"]) * .1)
                if train:
                    collected.append([prev, action, reward, state, False,
                                      (state["cycle"] - prev["cycle"]) * .1])
            action = learners.act(method, state, epsilon)
            if method == "Q-Learning" and tabular_key(state) not in learners.table:
                unseen_states += 1
            decisions.append(dict(cycle=state["cycle"], phase=state["global_phase"],
                                  green_cycles=state["green_cycles"], action=action,
                                  **{name: state[name] for name in EXTRA_FIELDS[:8]}))
            pending = (state, action)
            await env.set_action(action)
            await env.step_cycles(1)
        else:
            until_sample = 50 - state["cycle"] % 50
            await env.step_cycles(min(state["next_decision_in_cycles"], until_sample,
                                      horizon - state["cycle"]))
        state = await env.read_state()
        if state["cycle"] >= next_progress:
            progress = dict(method=method, seed=seed, training=train, cycle=state["cycle"],
                            horizon=horizon, passed=state["total_vehicles_passed"],
                            queue=state["current_queue_length"], updates=learners.updates,
                            elapsed_seconds=round(time.monotonic()-t0, 1))
            if env.output is not None:
                temp = env.output / f"progress_{env.port}.tmp"
                temp.write_text(json.dumps(progress, indent=2) + "\n")
                temp.replace(env.output / f"progress_{env.port}.json")
            print(f"Progress {method} seed={seed}: cycle {state['cycle']}/{horizon}, "
                  f"passed={state['total_vehicles_passed']:.0f}, queue={state['current_queue_length']:.0f}, "
                  f"updates={learners.updates}", flush=True)
            next_progress += 1000
        if state["cycle"] % 50 == 0:
            samples.append(dict(state))
    if pending is not None:
        prev, action = pending
        reward = -(state["queue_time_integral"] - prev["queue_time_integral"]) / 50.0
        total_reward += reward
        # Horizon is a truncation of continuing traffic, not an absorbing state.
        # Never bootstrap at an intermediate yellow/clearance/forced-left state.
        if state["dqn_waiting_action"]:
            if train and not collect_only:
                learners.learn(prev, action, reward, state, False, (state["cycle"] - prev["cycle"]) * .1)
            if train:
                collected.append([prev, action, reward, state, False, (state["cycle"] - prev["cycle"]) * .1])
    errors = [m for m in env._server_messages if isinstance(m, dict) and m.get("type") in
              ("SimulationError", "RuntimeError", "GamaServerError", "SimulationErrorDialog")]
    if errors:
        raise RuntimeError(f"GAMA runtime errors: {errors[:2]}")
    if not train and learners.policy_digest() != before_policy:
        raise RuntimeError("Evaluation modified a learned policy")
    created = state["total_vehicles_created"]
    if state["ground_requests"] != created + state["ground_pending"]:
        raise RuntimeError("Ground demand accounting failed")
    actions = Counter(d["action"] for d in decisions)
    metrics = dict(method=method, seed=seed, horizon=horizon, training=train, epsilon=epsilon,
                   throughput=state["total_vehicles_passed"], created=created,
                   requested=state["ground_requests"], external_queue=state["ground_pending"],
                   remaining=created-state["total_vehicles_passed"],
                   queue_mean=state["queue_time_integral"] / (horizon * .1),
                   stopped_vehicle_seconds=state["queue_time_integral"],
                   external_vehicle_seconds=state["external_wait_integral"],
                   accrued_wait_per_request=(state["queue_time_integral"] + state["external_wait_integral"]) /
                       max(1, state["ground_requests"]),
                   active_wait_mean=state["average_waiting_time"],
                   max_wait=state["cumulative_max_wait"],
                   completed_travel_mean=state["average_travel_time"],
                   speed_mean=float(np.mean([s["average_speed"] for s in samples])),
                   extend=actions[0], switch=actions[1], reward=total_reward,
                   unseen_q_states=unseen_states,
                   transitions=learners.transitions, updates=learners.updates,
                   seconds=round(time.monotonic()-t0, 2),
                   policy_sha256=before_policy, demand_sha256=demand_hash,
                   source_sha256=sources,
                   truncated_transition=bool(pending is not None and not state["dqn_waiting_action"]))
    output.mkdir(parents=True, exist_ok=True)
    if train:
        (output / "transitions.json").write_text(json.dumps(collected) + "\n")
    for name, rows in (("series", samples), ("decisions", decisions)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    # This file is the completion marker used when resuming cached episodes.
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{method} seed={seed} train={train} passed={metrics['throughput']:.0f} "
          f"queue={metrics['queue_mean']:.2f} EXTEND/SWITCH={actions[0]}/{actions[1]} "
          f"updates={learners.updates} wall={metrics['seconds']:.0f}s", flush=True)
    return metrics


def report(output):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((output / "test_results.json").read_text())
    df = pd.DataFrame(rows)
    expected = json.loads((output / "manifest.json").read_text())["test_seeds"]
    for method in METHODS:
        assert sorted(df.loc[df.method == method, "seed"].tolist()) == expected
    for _, paired in df.groupby("seed"):
        assert paired.demand_sha256.nunique() == paired.requested.nunique() == 1
    df.to_csv(output / "per_run_metrics.csv", index=False)
    metrics = [
        ("throughput", "Completed ground trips", "vehicles"),
        ("queue_mean", "Mean stopped vehicles in network", "vehicles"),
        ("accrued_wait_per_request", "Accrued wait / requested ground vehicle", "seconds"),
        ("external_queue", "Ground requests still outside network", "vehicles"),
        ("completed_travel_mean", "Travel time of completed ground trips", "seconds"),
        ("max_wait", "Maximum in-network cumulative wait", "seconds"),
    ]
    summary = df.groupby("method", sort=False)[[m[0] for m in metrics]].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    colors = ["#c64740", "#29835c", "#2879b8", "#7d50a5"]
    for ax, (key, title, unit) in zip(axes.flat, metrics):
        groups = [df.loc[df.method == method, key] for method in METHODS]
        means = [v.mean() for v in groups]
        stds = [v.std(ddof=1) if len(v) > 1 else 0 for v in groups]
        ax.bar(range(4), means, yerr=stds, color=colors, capsize=5, alpha=.8)
        for i, values in enumerate(groups):
            ax.scatter(np.full(len(values), i), values, color="black", s=16, zorder=3)
        ax.set_xticks(range(4), ["Fixed-Time", "MaxPressure", "Q-Learning", "DQN"], fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle(f"Paired evaluation: {len(expected)} held-out demand seeds; mean ± sample SD", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(output / "comparison.png", dpi=200)
    plt.close(fig)
    lines = ["# Đánh giá bốn phương pháp sau sửa lỗi", "",
             f"Mỗi phương pháp: {len(expected)} seed kiểm tra riêng, 8.000 cycle/seed. "
             "Mỗi seed dùng cùng một file yêu cầu xe cho cả bốn phương pháp. "
             "Q-Learning và DQN được đóng băng trong lúc đánh giá.", "",
             "| Chỉ số (trung bình ± độ lệch chuẩn mẫu) | Fixed-Time | MaxPressure | Q-Learning | DQN |",
             "|---|---:|---:|---:|---:|"]
    labels = ["Xe mặt đất hoàn thành", "Hàng đợi trong mạng trung bình (xe)",
              "Chờ đã tích lũy / yêu cầu xe (s)", "Yêu cầu còn đợi ngoài mạng (xe)",
              "Hành trình của xe đã hoàn thành (s)", "Chờ trong mạng lớn nhất (s)"]
    for label, (key, _, _) in zip(labels, metrics):
        cells = [f"{summary.loc[m, (key, 'mean')]:.2f} ± {summary.loc[m, (key, 'std')]:.2f}" for m in METHODS]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    winner = summary[("queue_mean", "mean")].idxmin()
    lines += ["", f"Theo mục tiêu chính giảm hàng đợi trong mạng, {winner} có trung bình thấp nhất trong tập kiểm tra này.",
              "", "Giới hạn: chỉ một seed huấn luyện mạng; số seed kiểm tra còn nhỏ. "
              "Độ lệch chuẩn phản ánh biến thiên giữa demand seed, không phải giữa các lần huấn luyện. "
              "Chờ / yêu cầu là chi phí đã tích lũy đến cuối mô phỏng, gồm cả hàng đợi ngoài mạng; "
              "thời gian di chuyển chỉ tính xe đã hoàn thành. Không suy ra thứ hạng phổ quát.",
              "", "Lịch đèn và quyết định dùng chung mô hình `intersection-benchmark.gaml`; "
              "Fixed-Time có xanh đi thẳng 30.1s vì cùng handshake 0.1s với các chính sách còn lại. "
              "Hai phương pháp học được huấn luyện off-policy trên cùng các transition; "
              "checkpoint được chọn bằng hai seed validation 3001–3002, tách khỏi test 5001 trở lên.",
              "", "Xem `manifest.json`, `training_log.json`, `validation/`, `per_run_metrics.csv` "
              "và `test/*/decisions.csv` để tái kiểm tra cấu hình, quá trình học và hành động."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(), flush=True)


async def pipeline(args, managers):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    learners = Learners(args.agent_seed)
    sources = source_hashes()
    manifest = dict(train_episodes=args.train_episodes, horizon=args.horizon, agent_seed=args.agent_seed,
                    validate_every=args.validate_every, workers=args.workers,
                    train_seeds=list(range(1001, 1001 + args.train_episodes)),
                    validation_seeds=[3001, 3002], test_seeds=list(range(5001, 5001+args.runs)),
                    code_sha256=sources,
                    episode_isolation="Fresh owned GAMA JVM per episode when --start-server is used",
                    training="Both off-policy learners see the same transition dataset; rollout policies frozen per worker batch",
                    checkpoint_selection="Minimum mean in-network queue on validation seeds",
                    reward="Negative integrated stopped vehicle seconds / 50; gamma=0.97**(dt/5)",
                    fixed_through_green="30.1 seconds including shared 0.1 second action handshake")
    if args.resume:
        old = json.loads((output / "manifest.json").read_text())
        if old != manifest:
            raise RuntimeError("Resume requires identical code, seeds and training configuration")
        learners.load_training(output / "last_checkpoint.pt")
    else:
        if (output / "training_log.json").exists():
            raise RuntimeError("Output already contains training; use --resume or a new directory")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    envs = [BenchmarkEnv(args.port + i, manager, output) for i, manager in enumerate(managers)]
    for env in envs:
        env.expected_sources = sources
    await asyncio.gather(*(env.setup() for env in envs))
    all_rows = json.loads((output / "training_log.json").read_text()) if args.resume else []

    async def run_jobs(jobs):
        # Snapshot policies before parallel collection. Merge transitions later
        # in seed order, so worker completion order cannot affect learning.
        actors = []
        for job in jobs:
            actor = Learners(args.agent_seed * 100000 + job["seed"])
            actor.online.load_state_dict(learners.online.state_dict())
            actor.table = copy.deepcopy(learners.table)
            actor.transitions, actor.updates = learners.transitions, learners.updates
            if job.get("checkpoint"):
                actor.load_policy(job["checkpoint"], job["method"])
            actors.append(actor)
        queue = asyncio.Queue()
        for index in range(len(jobs)):
            queue.put_nowait(index)
        results = [None] * len(jobs)

        async def worker(env):
            while not queue.empty():
                index = queue.get_nowait()
                job = jobs[index]
                results[index] = await run_episode(
                    env, actors[index], job["method"], job["seed"], args.horizon,
                    job["demand"], job["output"], job.get("train", False),
                    job.get("epsilon", 0.0), collect_only=job.get("train", False))
        tasks = [asyncio.create_task(worker(env)) for env in envs]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return results

    def job(method, seed, folder, **kwargs):
        demand = output / "demand" / f"{seed}.csv"
        generate_demand(demand, seed, args.horizon)
        return dict(method=method, seed=seed, demand=demand, output=folder, **kwargs)

    try:
        if args.smoke:
            seeds = [5001, 5001, 5002, 5003, 5001]
            all_rows = await run_jobs([job("Fixed-Time", seed, output / f"smoke_{i}")
                                       for i, seed in enumerate(seeds)])
            fields = [k for k in all_rows[0] if k not in ("seconds",)]
            assert all(all_rows[0][k] == all_rows[1][k] for k in fields), "Same demand/policy must repeat"
            assert all(all_rows[0][k] == all_rows[4][k] for k in fields), "Fresh JVM must repeat same episode"
            (output / "smoke_results.json").write_text(json.dumps(all_rows, indent=2) + "\n")
            return

        best = (json.loads((output / "validation_best.json").read_text())
                if args.resume and (output / "validation_best.json").exists()
                else {method: float("inf") for method in METHODS[2:]})
        best_paths = {method: output / ("best_q.pt" if method == "Q-Learning" else "best_dqn.pt")
                      for method in METHODS[2:]}
        async def validate(episode):
            validation = await run_jobs([
                job(candidate, seed, output / "validation" / f"ep{episode}_{candidate.replace(' ', '_')}_{seed}")
                for candidate in METHODS[2:] for seed in (3001, 3002)])
            for candidate in METHODS[2:]:
                score = float(np.mean([r["queue_mean"] for r in validation if r["method"] == candidate]))
                if score < best[candidate]:
                    best[candidate] = score
                    learners.save(best_paths[candidate])
                    print(f"Validation best {candidate}: queue_mean={score:.3f} at episode {episode}", flush=True)
            (output / "validation_best.json").write_text(json.dumps(best, indent=2) + "\n")

        if args.resume and all_rows and (len(all_rows) % args.validate_every == 0 or len(all_rows) == args.train_episodes):
            await validate(len(all_rows))
        while len(all_rows) < args.train_episodes:
            start = len(all_rows) + 1
            boundary = min(args.train_episodes, ((start - 1) // args.validate_every + 1) * args.validate_every)
            end = min(start + args.workers - 1, boundary)
            jobs = [job(METHODS[2 + episode % 2], 1000 + episode, output / "train" / f"{episode:03d}",
                        train=True, epsilon=max(.05, 1.0 - .95 * (episode - 1) / max(1, args.train_episodes * .8)))
                    for episode in range(start, end + 1)]
            collected = await run_jobs(jobs)
            for episode, spec, metrics in zip(range(start, end + 1), jobs, collected):
                for transition in json.loads((spec["output"] / "transitions.json").read_text()):
                    learners.learn(*transition)
                metrics.update(episode=episode, transitions=learners.transitions, updates=learners.updates)
                (spec["output"] / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
                all_rows.append(metrics)
                print(f"Learned episode {episode}/{args.train_episodes}: "
                      f"transitions={learners.transitions}, updates={learners.updates}", flush=True)
            learners.save(output / "last_checkpoint.pt")
            (output / "training_log.json").write_text(json.dumps(all_rows, indent=2) + "\n")
            if end == boundary:
                await validate(end)

        before = learners.policy_digest()
        results = await run_jobs([
            job(method, seed, output / "test" / f"{method.replace(' ', '_')}_{seed}",
                checkpoint=best_paths.get(method))
            for method in METHODS for seed in range(5001, 5001 + args.runs)])
        (output / "test_results.json").write_text(json.dumps(results, indent=2) + "\n")
        assert before == learners.policy_digest(), "Evaluation changed training policies"
        report(output)
    finally:
        await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=6876)
    p.add_argument("--horizon", type=int, default=8000)
    p.add_argument("--train-episodes", type=int, default=40)
    p.add_argument("--validate-every", type=int, default=10)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--agent-seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--start-server", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.horizon, args.train_episodes, args.validate_every, args.runs, args.workers) <= 0:
        p.error("Horizon, episodes, validation interval and runs must be positive")
    if not args.start_server and args.workers != 1:
        p.error("Multiple workers require --start-server to own isolated GAMA processes")
    managers = []
    try:
        if args.start_server:
            for i in range(args.workers):
                port = args.port + i
                manager = GamaServerManager(port=port, workdir=str(args.output / "server_ws" / str(port)),
                    log_file=str(args.output / "server_logs" / f"port{port}_episode_001.log"))
                managers.append(manager)
                manager.start(reuse_running=False)
        else:
            managers = [None]
        asyncio.run(pipeline(args, managers))
    finally:
        for manager in managers:
            if manager:
                manager.stop()


if __name__ == "__main__":
    main()
