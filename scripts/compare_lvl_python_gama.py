#!/usr/bin/env python3
"""Compare the GeoJSON-backed Le Van Luong Python model with GAMA."""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dqn_py"))

from gama_env import GamaTrafficEnv
from server_manager import GamaServerManager
from LeVanLuongIntersection.python_sim import LeVanLuongSimulator
from LeVanLuongIntersection.demand import generate_demand, demand_parameters


MODEL = ROOT / "LeVanLuongIntersection" / "models" / "LeVanLuong.gaml"
EXPR = ('"" + cycle + "|" + phase + "|" + length(vehicle) + "|" + '
        'total_vehicles_created + "|" + total_vehicles_passed + "|" + '
        'front_row_max["NE"] + "|" + front_row_max["SE"] + "|" + '
        'front_row_max["SW"] + "|" + front_row_max["NW"] + "|" + '
        'queue_time_integral + "|" + system_time_integral + "|" + '
        'weighted_delay_integral')
FIELDS = ("active", "created", "passed", "front_row_max_NE", "front_row_max_SE",
          "front_row_max_SW", "front_row_max_NW",
          "queue_time_integral", "system_time_integral", "weighted_delay_integral")


async def gama_snapshot(env):
    raw = str(await env._exec_expression(EXPR)).split("|")
    if len(raw) != 12:
        raise RuntimeError(f"Invalid Le Van Luong payload: {raw}")
    return {"cycle": int(float(raw[0])), "phase": raw[1],
            **{key: (int(float(value)) if i < 7 else round(float(value), 3))
               for i, (key, value) in enumerate(zip(FIELDS, raw[2:]))}}


async def main(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    demand = output.with_name(output.stem + "_demand.csv")
    demand_sha = generate_demand(demand, args.seed, args.horizon)
    python_sim = LeVanLuongSimulator(seed=args.seed, demand_file=demand)
    manager = None
    env = None
    python_wall = gama_wall = 0.0
    try:
        if args.start_server:
            manager = GamaServerManager(port=args.port,
                workdir=str(output.parent / "lvl_gama_ws"),
                log_file=str(output.parent / "lvl_gama_server.log"))
            manager.start(reuse_running=False)
        env = GamaTrafficEnv(port=args.port, model_path=str(MODEL),
                             experiment="python_parity", verbose=False)
        await env.setup()
        await env.load_experiment(parameters=[
            {"name": "traffic_seed", "type": "float", "value": str(float(args.seed))},
            *demand_parameters(demand)])
        rows = []
        while True:
            py = python_sim.snapshot()
            ga = await gama_snapshot(env)
            if py["cycle"] != ga["cycle"]:
                raise RuntimeError(f"Cycle mismatch: Python {py['cycle']}, GAMA {ga['cycle']}")
            row = {"cycle": py["cycle"], "python_phase": py["phase"],
                   "gama_phase": ga["phase"]}
            for key in FIELDS:
                row[f"python_{key}"] = py[key]
                row[f"gama_{key}"] = ga[key]
                row[f"absolute_error_{key}"] = abs(py[key] - ga[key])
            rows.append(row)
            if py["cycle"] >= args.horizon:
                break
            n = min(args.sample_every, args.horizon - py["cycle"])
            started = time.perf_counter()
            python_sim.step(n)
            python_wall += time.perf_counter() - started
            started = time.perf_counter()
            await env.step_cycles(n)
            gama_wall += time.perf_counter() - started
    finally:
        if manager is not None:
            manager.stop(force=True)
            if env is not None:
                try:
                    env.disconnect()
                except Exception:
                    pass
        elif env is not None:
            await env.close()
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"seed": args.seed, "horizon": args.horizon,
               "demand_sha256": demand_sha,
               "python_step_wall_seconds": round(python_wall, 3),
               "gama_step_wall_seconds": round(gama_wall, 3),
               "step_speedup": round(gama_wall / python_wall, 2) if python_wall else None,
               "phase_match": all(r["python_phase"] == r["gama_phase"] for r in rows),
               "final_active_error": rows[-1]["absolute_error_active"],
               "final_active_relative_error": rows[-1]["absolute_error_active"] /
                   max(1, rows[-1]["gama_active"]),
               "final_created_error": rows[-1]["absolute_error_created"],
               "final_passed_error": rows[-1]["absolute_error_passed"],
               "final_queue_time_integral_error": rows[-1]["absolute_error_queue_time_integral"],
               "final_queue_time_integral_relative_error": round(
                   rows[-1]["absolute_error_queue_time_integral"] /
                   max(1.0, rows[-1]["gama_queue_time_integral"]), 4),
               "final_system_time_integral_error": rows[-1]["absolute_error_system_time_integral"],
               "final_system_time_integral_relative_error": round(
                   rows[-1]["absolute_error_system_time_integral"] /
                   max(1.0, rows[-1]["gama_system_time_integral"]), 4),
               "final_weighted_delay_integral_error": rows[-1]["absolute_error_weighted_delay_integral"],
               "final_weighted_delay_integral_relative_error": round(
                   rows[-1]["absolute_error_weighted_delay_integral"] /
                   max(1.0, rows[-1]["gama_weighted_delay_integral"]), 4),
               "max_queue_time_integral_relative_error": round(max(
                   r["absolute_error_queue_time_integral"] / max(1.0, r["gama_queue_time_integral"])
                   for r in rows), 4),
               "max_system_time_integral_relative_error": round(max(
                   r["absolute_error_system_time_integral"] / max(1.0, r["gama_system_time_integral"])
                   for r in rows), 4),
               "max_weighted_delay_integral_relative_error": round(max(
                   r["absolute_error_weighted_delay_integral"] / max(1.0, r["gama_weighted_delay_integral"])
                   for r in rows), 4),
               "final_front_row_max_errors": {arm: rows[-1][f"absolute_error_front_row_max_{arm}"]
                                              for arm in ("NE", "SE", "SW", "NW")}}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.enforce:
        passed = (summary["phase_match"] and
                  summary["final_active_relative_error"] <= args.tolerance and
                  summary["final_passed_error"] <= max(5, int(args.horizon * 0.01)) and
                  summary["final_queue_time_integral_relative_error"] <= args.tolerance and
                  summary["final_system_time_integral_relative_error"] <= args.tolerance and
                  summary["final_weighted_delay_integral_relative_error"] <= args.tolerance)
        if not passed:
            raise SystemExit(f"Parity enforcement failed with tolerance {args.tolerance}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--port", type=int, default=6977)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Maximum relative error threshold for enforcement")
    parser.add_argument("--enforce", action="store_true",
                        help="Exit non-zero if parity thresholds are violated")
    parser.add_argument("--output", type=Path,
                        default=Path("comparison_results/lvl_python_gama.csv"))
    options = parser.parse_args()
    if options.horizon <= 0 or options.sample_every <= 0:
        parser.error("Horizon and sample interval must be positive")
    asyncio.run(main(options))
