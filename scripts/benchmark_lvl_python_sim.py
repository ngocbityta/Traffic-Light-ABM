#!/usr/bin/env python3
"""Measure Le Van Luong Python simulator and LVL training observation throughput.

Examples:
  python scripts/benchmark_lvl_python_sim.py --mode both --cycles 4000
  python scripts/benchmark_lvl_python_sim.py --mode train --cycles 4000 --profile

Cycles are simulation ticks (0.2 seconds each). Training steps are decision
transitions and include simulator stepping plus LVLPythonEnv.read_state's
spatial observation construction. No GAMA server or learner update is used.
"""
import argparse
import asyncio
import cProfile
import json
import pstats
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON_SIM_ROOT = ROOT / "python-simulation" / "simulation"
sys.path.insert(0, str(PYTHON_SIM_ROOT))
sys.path.insert(0, str(ROOT / "train" / "logic"))
from train.logic.lvl_env import LVLPythonEnv
from python_simulation.python_sim import ARMS, LeVanLuongSimulator, road_lengths
import csv
import random


def generate_lvl_demand(path, seed, horizon):
    """Create valid arm-indexed demand for the four-arm Python simulator."""
    rng = random.Random(seed)
    roads = road_lengths()
    rows = []
    for cycle in range(0, horizon, 15):
        for arm in range(4):
            if rng.random() < .45:
                is_car = int(rng.random() < .15)
                lane_count = roads[f"{ARMS[arm]}_in"]["num_lanes"]
                occupied = 2 if is_car else 1
                rows.append((cycle, arm, is_car, rng.uniform(7, 13),
                             rng.randrange(3), rng.randrange(lane_count - occupied + 1)))
    with Path(path).open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def fixed_run(cycles, seed, demand):
    sim = LeVanLuongSimulator(seed=seed, demand_file=demand, train_mode=False)
    start = time.perf_counter()
    sim.step(cycles)
    elapsed = time.perf_counter() - start
    return {"mode": "fixed", "cycles": cycles, "elapsed_sec": elapsed,
            "cycles_per_sec": cycles / elapsed if elapsed else None,
            "simulated_sec_per_wall_sec": cycles * .2 / elapsed if elapsed else None,
            "created": sim.created, "active_final": len(sim.vehicles),
            "passed": sim.passed}


async def train_run(cycles, seed, demand):
    env = LVLPythonEnv(state_mode="spatial", reward_mode="r1")
    state = await env.reset(seed=seed, parameters=[{"name": "demand_file_path",
                                                   "value": str(demand)}])
    start = time.perf_counter()
    env.sim.step(state["next_decision_in_cycles"])
    state = await env.read_state()
    decisions = 0
    while env.sim.cycle < cycles:
        if not state["dqn_waiting_action"]:
            raise RuntimeError(f"Expected decision tick, got {env.sim.cycle}")
        env.sim.set_action(0 if decisions % 2 == 0 else 2)
        env.sim.step(1)
        count = min(env.sim.next_decision_in_cycles, max(0, cycles - env.sim.cycle))
        if count:
            env.sim.step(count)
        state = await env.read_state()
        decisions += 1
    elapsed = time.perf_counter() - start
    return {"mode": "train", "requested_horizon_cycles": cycles,
            "final_cycle": env.sim.cycle, "decision_steps": decisions,
            "elapsed_sec": elapsed,
            "decision_steps_per_sec": decisions / elapsed if elapsed else None,
            "cycles_per_sec": env.sim.cycle / elapsed if elapsed else None,
            "created": env.sim.created, "active_final": len(env.sim.vehicles),
            "passed": env.sim.passed}


def profile(fn, label, output_path, limit):
    profiler = cProfile.Profile()
    result = profiler.runcall(fn)
    profiler.dump_stats(str(output_path))
    print(f"\n[{label} profile: top {limit} by cumulative time]")
    pstats.Stats(profiler, stream=sys.stdout).sort_stats("cumulative").print_stats(limit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "train", "both"), default="both")
    parser.add_argument("--cycles", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--demand", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "reports" / "lvl_sim_profiles")
    parser.add_argument("--profile-top", type=int, default=25)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    temp = None
    demand = args.demand
    if demand is None:
        temp = tempfile.TemporaryDirectory(prefix="lvl_perf_")
        demand = Path(temp.name) / "demand.csv"
        generate_lvl_demand(demand, args.seed, args.cycles + 1)
    outputs = []
    try:
        modes = ("fixed", "train") if args.mode == "both" else (args.mode,)
        for mode in modes:
            if mode == "fixed":
                fn = lambda: fixed_run(args.cycles, args.seed, demand)
                call = fn
            else:
                call = lambda: asyncio.run(train_run(args.cycles, args.seed, demand))
            if args.profile:
                args.profile_dir.mkdir(parents=True, exist_ok=True)
                result = profile(call, mode, args.profile_dir / f"{mode}.prof", args.profile_top)
            else:
                result = call()
            outputs.append(result)
        print(json.dumps({"python": sys.version.split()[0], "seed": args.seed,
                          "demand": str(demand), "results": outputs}, indent=2))
    finally:
        if temp:
            temp.cleanup()


if __name__ == "__main__":
    main()
