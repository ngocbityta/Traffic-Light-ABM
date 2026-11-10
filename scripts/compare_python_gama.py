#!/usr/bin/env python3
"""Run paired demand and a fixed four-phase policy on GAMA and Python.

The report contains raw series and absolute/relative errors.  The Python
engine has no scientific parity claim until this comparison is satisfactory
for the intended seeds, demand, policy, horizon, and output metrics.
"""

import argparse
import asyncio
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dqn_py"))

from demand import demand_parameters, generate_demand
from gama_env import GamaTrafficEnv
from python_env import PythonTrafficEnv
from server_manager import GamaServerManager


FIELDS = ("global_phase", "decision_count", "green_cycles",
          "total_vehicles_created", "total_vehicles_passed", "current_queue_length",
          "average_waiting_time", "average_speed", "average_travel_time",
          "queue_time_integral", "external_wait_integral")
PARITY_FIELDS = ("total_vehicles_created", "total_vehicles_passed",
                 "current_queue_length", "average_waiting_time", "average_speed",
                 "average_travel_time", "queue_time_integral", "external_wait_integral")
SPATIAL_FIELDS = FIELDS[:-2] + ("system_time_integral", "stopped_time_integral",
                               "external_all_time_integral", "outgoing_block_integral",
                               "weighted_delay_integral")


async def run(env, seed, params, horizon, sample_every):
    await env.setup()
    state = await env.reset(seed=seed, parameters=params)
    samples = []
    fields = SPATIAL_FIELDS if env.state_mode == "spatial" else FIELDS
    required = "spatial" if env.state_mode == "spatial" else "queue_time_integral"
    while state["cycle"] <= horizon:
        if required not in state:
            state = await env.read_state(force_dtse=True)
        sample = {"cycle": state["cycle"], **{k: state[k] for k in fields}}
        if env.state_mode == "spatial":
            sample["_observation"] = state["spatial"].copy()
            sample["_pressure"] = state["pressure"].copy()
            sample["_cost"] = env.reward_cost_from_state(state)
        samples.append(sample)
        if state["cycle"] == horizon:
            break
        end = min(horizon, (state["cycle"] // sample_every + 1) * sample_every)
        while state["cycle"] < end:
            if state["dqn_waiting_action"]:
                await env.set_action(state["decision_count"] % 4)
                await env.step_cycles(1)
            else:
                await env.step_cycles(min(end - state["cycle"], state["next_decision_in_cycles"]))
            state = await env.read_state()
    return samples


async def main(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paired_demand_") as td:
        demand = Path(td) / "demand.csv"
        demand_sha = generate_demand(demand, args.seed, args.horizon)
        params = demand_parameters(demand)
        py = PythonTrafficEnv(state_mode=args.state_mode,
                              reward_mode=args.reward_mode, verbose=False)
        python_samples = await run(py, args.seed, params, args.horizon, args.sample_every)
        manager = None
        gama = None
        try:
            if args.start_server:
                manager = GamaServerManager(port=args.port,
                    workdir=str(output.parent / "gama_ws"),
                    log_file=str(output.parent / "gama_server.log"))
                manager.start(reuse_running=False)
            gama = GamaTrafficEnv(port=args.port, state_mode=args.state_mode,
                                  reward_mode=args.reward_mode, verbose=False,
                                  fast_training=True, reset_mode="fresh")
            gama_samples = await run(gama, args.seed, params, args.horizon, args.sample_every)
        finally:
            if manager is not None:
                manager.stop(force=True)
                if gama is not None:
                    try:
                        gama.disconnect()
                    except Exception:
                        # The JVM has already closed its socket; preserve any
                        # simulation error raised above.
                        pass
            elif gama is not None:
                await gama.close()
    rows = []
    fields = SPATIAL_FIELDS if args.state_mode == "spatial" else FIELDS
    parity_fields = (tuple(k for k in PARITY_FIELDS if k in fields) +
                     (("system_time_integral",) if args.state_mode == "spatial" else ()))
    for a, b in zip(python_samples, gama_samples):
        if a["cycle"] != b["cycle"]:
            raise RuntimeError(f"Sample cycle mismatch: {a['cycle']} vs {b['cycle']}")
        row = {"cycle": a["cycle"]}
        for key in fields:
            row[f"python_{key}"] = a[key]
            row[f"gama_{key}"] = b[key]
            row[f"absolute_error_{key}"] = abs(a[key] - b[key])
            row[f"relative_error_{key}"] = abs(a[key] - b[key]) / max(1.0, abs(b[key]))
        if args.state_mode == "spatial":
            delta = np.abs(a["_observation"] - b["_observation"])
            row["spatial_mae"] = float(delta.mean())
            row["spatial_max_abs_error"] = float(delta.max())
            row["pressure_mae"] = float(np.abs(a["_pressure"] - b["_pressure"]).mean())
            row["python_reward_cost"] = a["_cost"]
            row["gama_reward_cost"] = b["_cost"]
            row["reward_cost_relative_error"] = abs(a["_cost"] - b["_cost"]) / max(1, abs(b["_cost"]))
        rows.append(row)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"seed": args.seed, "horizon": args.horizon,
               "sample_every": args.sample_every, "demand_sha256": demand_sha,
               "python_fidelity": py.fidelity,
               "final_relative_errors": {key: rows[-1][f"relative_error_{key}"] for key in fields},
               "mean_relative_errors": {key: sum(r[f"relative_error_{key}"] for r in rows) / len(rows)
                                        for key in fields}}
    summary["tolerance"] = args.tolerance
    summary["state_mode"] = args.state_mode
    summary["reward_mode"] = args.reward_mode
    if args.state_mode == "spatial":
        summary["mean_spatial_mae"] = sum(r["spatial_mae"] for r in rows) / len(rows)
        summary["mean_reward_cost_relative_error"] = sum(
            r["reward_cost_relative_error"] for r in rows) / len(rows)
    summary["sampled_phase_match"] = all(
        row["python_global_phase"] == row["gama_global_phase"] and
        row["python_decision_count"] == row["gama_decision_count"]
        for row in rows)
    summary["within_tolerance"] = (summary["sampled_phase_match"] and all(
        rows[-1][f"relative_error_{key}"] <= args.tolerance and
        summary["mean_relative_errors"][key] <= args.tolerance
        for key in parity_fields if abs(rows[-1][f"gama_{key}"]) >= 1))
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary["within_tolerance"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--port", type=int, default=6976)
    parser.add_argument("--tolerance", type=float, default=.1,
                        help="Maximum allowed final relative error for nonzero metrics")
    parser.add_argument("--enforce", action="store_true",
                        help="Exit with status 2 if final traffic metrics exceed tolerance")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--state-mode", choices=("dtse", "spatial"), default="dtse")
    parser.add_argument("--reward-mode", choices=("total_wait", "r1"), default="total_wait")
    parser.add_argument("--output", default="comparison_results/python_gama_parity.csv")
    options = parser.parse_args()
    if options.horizon <= 0 or options.sample_every <= 0 or options.tolerance < 0:
        parser.error("Horizon and sample interval must be positive")
    if (options.state_mode == "spatial") != (options.reward_mode == "r1"):
        parser.error("Spatial state requires r1; DTSE requires total_wait")
    result = asyncio.run(main(options))
    if options.enforce and not result:
        raise SystemExit(2)
