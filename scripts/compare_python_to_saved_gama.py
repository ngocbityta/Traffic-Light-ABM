#!/usr/bin/env python3
"""Recompare a changed Python simulator against recorded paired GAMA snapshots."""

import argparse
import asyncio
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train" / "logic"))

from train.logic.demand import demand_parameters, generate_demand
from train.logic.python_env import PythonTrafficEnv
from compare_python_gama import SPATIAL_FIELDS, run as run_dqn

FIELDS = ("global_phase", "total_vehicles_created", "total_vehicles_passed",
          "current_queue_length", "average_waiting_time", "average_speed",
          "average_travel_time", "queue_time_integral", "external_wait_integral")


async def main(args):
    reference = args.reference.resolve()
    metadata = json.loads(reference.with_suffix(".json").read_text())
    snapshots = list(csv.DictReader(reference.open()))
    if not snapshots or int(snapshots[0]["cycle"]) != 0:
        raise ValueError("Reference must contain a cycle-zero snapshot")
    horizon = metadata["horizon"]
    dqn_spatial = metadata.get("state_mode") == "spatial"
    fields = SPATIAL_FIELDS[1:] if dqn_spatial else FIELDS[1:]
    if int(snapshots[-1]["cycle"]) != horizon:
        raise ValueError("Reference horizon does not match its CSV")
    with tempfile.TemporaryDirectory(prefix="saved_parity_") as td:
        demand = Path(td) / "demand.csv"
        digest = generate_demand(demand, metadata["seed"], horizon)
        if digest != metadata["demand_sha256"]:
            raise ValueError("Demand hash differs from the paired GAMA run")
        env = PythonTrafficEnv(controller_mode="dqn" if dqn_spatial else metadata["controller"],
                               state_mode="spatial", reward_mode="r1", verbose=False)
        if dqn_spatial:
            started = time.perf_counter()
            python_samples = await run_dqn(env, metadata["seed"],
                                           demand_parameters(demand), horizon,
                                           metadata["sample_every"])
            step_seconds = time.perf_counter() - started
        else:
            await env.reset(seed=metadata["seed"], parameters=demand_parameters(demand))
        rows = []
        if not dqn_spatial:
            step_seconds = 0.0
        for i, snapshot in enumerate(snapshots):
            target = int(snapshot["cycle"])
            if dqn_spatial:
                state = python_samples[i]
                if state["cycle"] != target:
                    raise ValueError("Reference cycle differs from Python replay")
            else:
                if target < env._cycle:
                    raise ValueError("Reference cycles must be increasing")
                started = time.perf_counter()
                await env.step_cycles(target - env._cycle)
                step_seconds += time.perf_counter() - started
                state = await env.read_state(force_dtse=True)
            row = {"cycle": target}
            for field in ("global_phase",) + tuple(fields):
                py = state[field]
                ga = float(snapshot[f"gama_{field}"])
                row[f"python_{field}"] = py
                row[f"gama_{field}"] = ga
                row[f"relative_error_{field}"] = abs(py-ga)/max(1.0, abs(ga))
            rows.append(row)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = dict(reference=str(reference), controller="dqn" if dqn_spatial else metadata["controller"],
                  seed=metadata["seed"], horizon=horizon, demand_sha256=digest,
                  python_step_seconds=round(step_seconds, 3),
                  phase_mismatch_cycles=[r["cycle"] for r in rows if
                                         r["python_global_phase"] != r["gama_global_phase"]],
                  final_relative_errors={f: rows[-1][f"relative_error_{f}"] for f in fields},
                  mean_relative_errors={f: sum(r[f"relative_error_{f}"] for r in rows)/len(rows)
                                        for f in fields})
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
