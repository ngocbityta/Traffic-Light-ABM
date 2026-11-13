#!/usr/bin/env python3
"""Measure parallel LVL Python training rollouts on paired official demand.

Each process runs one complete, independent train-mode 8,000-tick episode
with the same demand, seed, and deterministic alternating legal action stream.
The timing includes environment reset, demand parsing, observations, and rollout;
the outer wall time also includes process-pool startup.
"""
import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import multiprocessing
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train" / "logic"))
from train.logic.lvl_env import LVLPythonEnv


def rollout_worker(task):
    worker_id, seed, horizon, demand_path, sample_every = task

    async def rollout():
        env = LVLPythonEnv(state_mode="spatial", reward_mode="r1")
        started = time.perf_counter()
        state = await env.reset(seed=seed, parameters=[
            {"name": "demand_file_path", "value": demand_path}])
        # Match the training protocol: first policy decision is at cycle 50.
        await env.step_cycles(state["next_decision_in_cycles"])
        state = await env.read_state()
        snapshots = []
        next_sample = sample_every
        decision = 0
        while state["cycle"] < horizon:
            if not state["dqn_waiting_action"]:
                raise RuntimeError(f"worker {worker_id}: not at decision tick {state['cycle']}")
            action = 0 if decision % 2 == 0 else 2
            state, _, _, _ = await env.step(action, horizon=horizon,
                                            current_state=state)
            decision += 1
            while next_sample <= state["cycle"]:
                if next_sample != state["cycle"]:
                    raise RuntimeError(
                        f"worker {worker_id}: crossed snapshot cycle {next_sample} "
                        f"and reached {state['cycle']}")
                snapshots.append(env.sim.snapshot())
                next_sample += sample_every
        elapsed = time.perf_counter() - started
        return {"worker": worker_id, "seed": seed, "final_cycle": env.sim.cycle,
                "decisions": decision, "elapsed_sec": elapsed,
                "created": env.sim.created, "passed": env.sim.passed,
                "active_final": len(env.sim.vehicles), "snapshots": snapshots}

    return asyncio.run(rollout())


def run_batch(workers, repeat, seed, horizon, demand, sample_every):
    tasks = [(worker, seed, horizon, str(demand), sample_every)
             for worker in range(workers)]
    started = time.perf_counter()
    # Explicit spawn keeps the benchmark behavior consistent on macOS and Linux.
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(rollout_worker, tasks))
    wall = time.perf_counter() - started
    signatures = [hashlib.sha256(json.dumps(
        row["snapshots"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for row in results]
    deterministic = all(sig == signatures[0] for sig in signatures)
    if not deterministic:
        raise AssertionError(f"worker snapshots diverged: {signatures}")
    if any(row["final_cycle"] != horizon for row in results):
        raise AssertionError("worker did not reach requested horizon")
    return {"repeat": repeat, "workers": workers, "wall_sec": wall,
            "episodes": len(results), "episodes_per_sec": len(results) / wall,
            "ticks_per_sec_aggregate": len(results) * horizon / wall,
            "deterministic_snapshots_across_workers": deterministic,
            "snapshot_count_per_worker": len(results[0]["snapshots"]),
            "snapshot_sha256": signatures[0],
            "worker_elapsed_sec": [row["elapsed_sec"] for row in results],
            "worker_final": [{k: row[k] for k in
                              ("worker", "final_cycle", "decisions", "created",
                               "passed", "active_final")} for row in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand", type=Path, default=ROOT /
        "reports/python_parity_20261110/lvl_queue_crossroad_8000_seed1_demand.csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=8000)
    parser.add_argument("--sample-every", type=int, default=400)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.demand.is_file():
        parser.error(f"demand file not found: {args.demand}")
    if (args.horizon <= 0 or args.sample_every <= 0 or
            args.horizon % args.sample_every or args.repeats <= 0 or
            any(count <= 0 for count in args.worker_counts)):
        parser.error("horizon/sample interval must be positive and divide evenly; counts/repeats positive")
    # Sequential batches avoid overlapping worker-count measurements.
    records = []
    for repeat in range(1, args.repeats + 1):
        for count in args.worker_counts:
            records.append(run_batch(count, repeat, args.seed, args.horizon,
                                     args.demand.resolve(), args.sample_every))
    grouped = {}
    for row in records:
        grouped.setdefault(str(row["workers"]), []).append(row["episodes_per_sec"])
    result = {"python": sys.version.split()[0], "seed": args.seed,
              "horizon": args.horizon, "sample_every": args.sample_every,
              "demand": str(args.demand.resolve()),
              "demand_sha256": hashlib.sha256(args.demand.read_bytes()).hexdigest(),
              "records": records,
              "median_episodes_per_sec": {k: statistics.median(v)
                                          for k, v in grouped.items()}}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
