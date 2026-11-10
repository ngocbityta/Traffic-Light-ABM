#!/usr/bin/env python3
"""Compare Python Fixed-Time/MaxPressure against their GAML models."""

import argparse
import asyncio
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dqn_py"))

from demand import demand_parameters, generate_demand
from gama_env import GamaTrafficEnv
from python_env import PythonTrafficEnv
from server_manager import GamaServerManager


MODEL = {"fixed": "intersection-fixed-time.gaml",
         "maxpressure": "intersection-maxpressure.gaml"}
FIELDS = ("cycle", "global_phase", "total_vehicles_created",
          "total_vehicles_passed", "current_queue_length",
          "average_waiting_time", "average_speed", "average_travel_time",
          "queue_time_integral", "external_wait_integral")
EXPR = '"" + ' + ' + "|" + '.join(FIELDS)


async def gama_snapshot(env):
    parts = str(await env._exec_expression(EXPR)).split("|")
    if len(parts) != len(FIELDS):
        raise RuntimeError(f"Invalid GAMA baseline metrics: {parts}")
    return {name: int(float(value)) if name in FIELDS[:5] else float(value)
            for name, value in zip(FIELDS, parts)}


async def main(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="baseline_demand_") as td:
        demand = Path(td) / "demand.csv"
        digest = generate_demand(demand, args.seed, args.horizon)
        python = PythonTrafficEnv(controller_mode=args.controller,
                                  state_mode="spatial", reward_mode="r1", verbose=False)
        await python.reset(seed=args.seed, parameters=demand_parameters(demand))
        python_step_seconds = 0.0
        gama_step_seconds = 0.0
        manager = None
        gama = None
        try:
            if args.start_server:
                manager = GamaServerManager(port=args.port,
                    workdir=str(output.parent / "baseline_gama_ws"),
                    log_file=str(output.parent / "baseline_gama_server.log"))
                manager.start(reuse_running=False)
            gama = GamaTrafficEnv(port=args.port,
                model_path=str(ROOT / "Traffic" / "models" / MODEL[args.controller]),
                experiment="python_parity", verbose=False)
            await gama.setup()
            await gama.load_experiment(parameters=[
                *demand_parameters(demand),
                {"name": "traffic_seed", "type": "float", "value": str(float(args.seed))},
                {"name": "fast_training", "type": "bool", "value": "true"}])
            rows = []
            while True:
                py = await python.read_state(force_dtse=True)
                ga = await gama_snapshot(gama)
                if py["cycle"] != ga["cycle"]:
                    raise RuntimeError(f"Cycle mismatch: {py['cycle']} vs {ga['cycle']}")
                row = {"cycle": py["cycle"]}
                for key in FIELDS[1:]:
                    row[f"python_{key}"] = py[key]
                    row[f"gama_{key}"] = ga[key]
                    row[f"relative_error_{key}"] = abs(py[key] - ga[key]) / max(1, abs(ga[key]))
                rows.append(row)
                if py["cycle"] >= args.horizon:
                    break
                n = min(args.sample_every, args.horizon - py["cycle"])
                started = time.perf_counter()
                await python.step_cycles(n)
                python_step_seconds += time.perf_counter() - started
                started = time.perf_counter()
                await gama.step_cycles(n)
                gama_step_seconds += time.perf_counter() - started
        finally:
            if manager is not None:
                manager.stop(force=True)
                if gama is not None:
                    try:
                        gama.disconnect()
                    except Exception:
                        pass
            elif gama is not None:
                await gama.close()
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"controller": args.controller, "seed": args.seed,
               "horizon": args.horizon, "demand_sha256": digest,
               "python_step_seconds": round(python_step_seconds, 3),
               "gama_step_seconds": round(gama_step_seconds, 3),
               "phase_match": all(r["python_global_phase"] == r["gama_global_phase"] for r in rows),
               "phase_mismatch_cycles": [r["cycle"] for r in rows
                                          if r["python_global_phase"] != r["gama_global_phase"]],
               "final_relative_errors": {key: rows[-1][f"relative_error_{key}"]
                                         for key in FIELDS[2:]},
               "mean_relative_errors": {key: sum(r[f"relative_error_{key}"] for r in rows) / len(rows)
                                        for key in FIELDS[2:]}}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=tuple(MODEL), required=True)
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--port", type=int, default=6978)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("comparison_results/python_baseline_parity.csv"))
    options = parser.parse_args()
    if options.horizon <= 0 or options.sample_every <= 0:
        parser.error("Horizon and sample interval must be positive")
    asyncio.run(main(options))
