#!/usr/bin/env python3
"""Run paired-demand Fixed-Time or MaxPressure without a GAMA process."""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dqn_py"))

from demand import demand_parameters, generate_demand
from python_env import PythonTrafficEnv


FIELDS = ("cycle", "global_phase", "total_vehicles_created",
          "total_vehicles_passed", "current_queue_length", "average_waiting_time",
          "max_waiting_time", "average_speed", "average_travel_time",
          "queue_time_integral", "external_wait_integral", "system_time_integral",
          "stopped_time_integral", "outgoing_block_integral", "weighted_delay_integral")


async def run(args):
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    demand = destination.with_suffix(".demand.csv")
    digest = generate_demand(demand, args.seed, args.horizon)
    env = PythonTrafficEnv(controller_mode=args.controller, state_mode="spatial",
                           reward_mode="r1", verbose=False)
    await env.setup()
    await env.reset(seed=args.seed, parameters=demand_parameters(demand))
    state = await env.read_state(force_dtse=True)
    start = time.perf_counter()
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({k: state[k] for k in FIELDS})
        while state["cycle"] < args.horizon:
            await env.step_cycles(min(args.sample_every, args.horizon - state["cycle"]))
            state = await env.read_state(force_dtse=True)
            writer.writerow({k: state[k] for k in FIELDS})
    summary = {"controller": args.controller, "seed": args.seed,
               "horizon": args.horizon, "demand_sha256": digest,
               "fidelity": env.fidelity, "wall_seconds": round(time.perf_counter() - start, 3),
               "final": {k: state[k] for k in FIELDS}}
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("fixed", "maxpressure"), required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--horizon", type=int, default=8000)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.horizon <= 0 or options.sample_every <= 0:
        parser.error("Horizon and sample interval must be positive")
    asyncio.run(run(options))
