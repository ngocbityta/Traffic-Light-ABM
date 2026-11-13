#!/usr/bin/env python3
"""Compare the GeoJSON-backed Le Van Luong Python model with GAMA."""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON_SIM_ROOT = ROOT / "python-simulation" / "simulation"
sys.path.insert(0, str(PYTHON_SIM_ROOT))
sys.path.insert(0, str(ROOT / "train" / "logic"))

from train.logic.gama_env import GamaTrafficEnv
from train.logic.server_manager import GamaServerManager
from python_simulation.python_sim import LeVanLuongSimulator, road_lengths
from python_simulation.demand import generate_demand, demand_parameters


MODEL = ROOT / "gama-simulation" / "simulation" / "models" / "LeVanLuong.gaml"
EXPR = ('"" + cycle + "|" + phase + "|" + length(vehicle) + "|" + '
        'total_vehicles_created + "|" + total_vehicles_passed + "|" + '
        'front_row_max["NE"] + "|" + front_row_max["SE"] + "|" + '
        'front_row_max["SW"] + "|" + front_row_max["NW"] + "|" + '
        'queue_time_integral + "|" + system_time_integral + "|" + '
        'weighted_delay_integral')
FIELDS = ("active", "created", "passed", "front_row_max_NE", "front_row_max_SE",
          "front_row_max_SW", "front_row_max_NW",
          "queue_time_integral", "system_time_integral", "weighted_delay_integral")
VEHICLES_EXPR = ('"" + (vehicle collect ("" + each.demand_row_id + ":" + '
                 'each.speed + ":" + each.lowest_lane + ":" + '
                 'each.num_lanes_occupied + ":" + '
                 '(each.current_road = nil ? "nil" : road(each.current_road).rid) + ":" + '
                 'each.distance_to_current_target))')
VEHICLE_RE = re.compile(
    r"(-?\d+):([\d.Ee+-]+):(-?\d+):(\d+):([A-Za-z0-9_]+|nil):([\d.Ee+-]+)"
)
def parse_gama_vehicles(raw, roads):
    """Decode GAMA's compact vehicle payload into demand-row keyed states."""
    states = {}
    for match in VEHICLE_RE.finditer(raw):
        row_id, speed, lane, occupied, road, remaining = match.groups()
        road = None if road == "nil" else road
        states[int(row_id)] = {"road": road,
                               "speed_mps": float(speed), "lane": int(lane),
                               "occupied": int(occupied),
                               "remaining_m": float(remaining),
                               "position_m": (roads[road]["length"] - float(remaining)
                                              if road in roads else None)}
    return states


def python_vehicle_states(simulator):
    """Expose roadless GAMA spawn attempts as ``current_road = nil``."""
    states = {}
    for vehicle in simulator.vehicles:
        if simulator.mode == "gama_2025_6_4" and not vehicle.gama_entered:
            road, position = None, None
        else:
            road, position = simulator._road_state(vehicle)
        states[vehicle.demand_row_id] = {
            "road": road, "lane": vehicle.lane, "occupied": vehicle.occupied,
            "speed_mps": vehicle.speed, "position_m": position}
    return states


def compare_vehicle_states(gama, python, position_tolerance, speed_tolerance):
    """Return per-vehicle discrepancies and the first structural/numeric split."""
    by_id = {}
    first = None
    for row_id in sorted(set(gama) | set(python)):
        gv, pv = gama.get(row_id), python.get(row_id)
        if gv is None or pv is None:
            delta = {"kind": "presence", "demand_row_id": row_id}
        elif gv["road"] != pv["road"]:
            delta = {"kind": "road", "demand_row_id": row_id,
                     "gama": gv["road"], "python": pv["road"]}
        elif gv["lane"] != pv["lane"]:
            delta = {"kind": "lane", "demand_row_id": row_id,
                     "gama": gv["lane"], "python": pv["lane"]}
        else:
            speed_error = abs(gv["speed_mps"] - pv["speed_mps"])
            position_error = (None if gv.get("position_m") is None or
                              pv.get("position_m") is None else
                              abs(gv["position_m"] - pv["position_m"]))
            if (speed_error > speed_tolerance or
                    (position_error is not None and position_error > position_tolerance)):
                delta = {"kind": "numeric", "demand_row_id": row_id,
                         "speed_error_mps": speed_error,
                         "position_error_m": position_error}
            else:
                delta = None
        if delta is not None:
            delta["gama_state"] = gv
            delta["python_state"] = pv
            by_id[row_id] = delta
            if first is None:
                first = delta
    return by_id, first


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
    source_hashes = {
        "python_sim_sha256": hashlib.sha256(
            (ROOT / "python-simulation/simulation/python_simulation/python_sim.py").read_bytes()).hexdigest(),
        "gama_model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "traffic_gama_model_sha256": hashlib.sha256(
            (ROOT / "gama-simulation/simulation/models/LeVanLuong.gaml").read_bytes()).hexdigest(),
        "demand_source_sha256": hashlib.sha256(
            (ROOT / "python-simulation/simulation/python_simulation/demand.py").read_bytes()).hexdigest(),
        "python_2025_transition_sha256": hashlib.sha256(
            (ROOT / "python-simulation/simulation/python_simulation/gama_2025_transition.py").read_bytes()).hexdigest(),
    }
    runtime = {"version": "2025.6.4", "headless": str(args.gama_headless),
               "traffic_jar": str(args.traffic_jar),
               "headless_sha256": hashlib.sha256(args.gama_headless.read_bytes()).hexdigest(),
               "traffic_jar_sha256": hashlib.sha256(args.traffic_jar.read_bytes()).hexdigest()}
    python_sim = LeVanLuongSimulator(seed=args.seed, demand_file=demand,
                                    mode=args.python_mode)
    roads = road_lengths()
    manager = None
    env = None
    python_wall = gama_wall = 0.0
    try:
        if args.start_server:
            manager = GamaServerManager(port=args.port,
                workdir=str(output.parent / "lvl_gama_ws"),
                log_file=str(output.parent / "lvl_gama_server.log"),
                gama_headless=str(args.gama_headless))
            manager.start(reuse_running=False)
        env = GamaTrafficEnv(port=args.port, model_path=str(MODEL),
                             experiment="python_parity", verbose=False)
        await env.setup()
        await env.load_experiment(parameters=[
            {"name": "traffic_seed", "type": "float", "value": str(float(args.seed))},
            {"name": "green_lvl", "type": "float", "value": "50.0"},
            {"name": "green_hdt", "type": "float", "value": "40.0"},
            {"name": "all_red", "type": "float", "value": "2.0"},
            {"name": "flow_per_approach", "type": "float", "value": "2400.0"},
            {"name": "motorbike_ratio", "type": "float", "value": "0.85"},
            {"name": "p_left", "type": "float", "value": "0.2"},
            {"name": "p_right", "type": "float", "value": "0.2"},
            *demand_parameters(demand)])
        rows = []
        first_divergence = None
        while True:
            py = python_sim.snapshot()
            ga = await gama_snapshot(env)
            if py["cycle"] != ga["cycle"]:
                raise RuntimeError(f"Cycle mismatch: Python {py['cycle']}, GAMA {ga['cycle']}")
            row = {"cycle": py["cycle"], "python_phase": py["phase"],
                   "gama_phase": ga["phase"]}
            gama_vehicles = parse_gama_vehicles(
                str(await env._exec_expression(VEHICLES_EXPR)), roads)
            python_vehicles = python_vehicle_states(python_sim)
            discrepancies, first_at_cycle = compare_vehicle_states(
                gama_vehicles, python_vehicles, args.position_tolerance,
                args.speed_tolerance)
            row["structural_divergences"] = sum(
                delta["kind"] in ("presence", "road", "lane")
                for delta in discrepancies.values())
            row["numeric_divergences"] = sum(
                delta["kind"] == "numeric" for delta in discrepancies.values())
            if first_at_cycle is not None and first_divergence is None:
                first_divergence = {"cycle": py["cycle"], **first_at_cycle,
                                    "gama_state": gama_vehicles.get(
                                        first_at_cycle["demand_row_id"]),
                                    "python_state": python_vehicles.get(
                                        first_at_cycle["demand_row_id"])}
            row["first_divergence"] = (json.dumps(first_at_cycle, sort_keys=True)
                                        if first_at_cycle else "")
            row["vehicle_divergences"] = json.dumps(
                list(discrepancies.values()), sort_keys=True)
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
               "python_mode": args.python_mode,
               "gama_runtime": runtime,
               "vehicle_gate_scope": "every_tick",
               "command": [sys.executable, *sys.argv],
               "demand_parameters": {"flow_per_approach": 2400.0,
                                     "motorbike_ratio": 0.85,
                                     "p_left": 0.2, "p_right": 0.2},
               "fixed_signal_action": {"mode": "fixed_time", "green_lvl_seconds": 50,
                                       "green_hdt_seconds": 40, "all_red_seconds": 3},
               "demand_sha256": demand_sha,
               "demand_csv": str(demand),
               "first_divergence": first_divergence,
               **source_hashes,
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
    checks = {
        "phase_match": summary["phase_match"],
        "no_vehicle_divergence": first_divergence is None,
        "final_active": summary["final_active_relative_error"] <= args.tolerance,
        "final_passed": summary["final_passed_error"] <= max(5, int(args.horizon * 0.01)),
        "final_queue_integral": summary["final_queue_time_integral_relative_error"] <= args.tolerance,
        "final_system_integral": summary["final_system_time_integral_relative_error"] <= args.tolerance,
        "final_weighted_delay_integral": summary["final_weighted_delay_integral_relative_error"] <= args.tolerance,
        "max_queue_integral": summary["max_queue_time_integral_relative_error"] <= args.max_intermediate_tolerance,
        "max_system_integral": summary["max_system_time_integral_relative_error"] <= args.max_intermediate_tolerance,
        "max_weighted_delay_integral": summary["max_weighted_delay_integral_relative_error"] <= args.max_intermediate_tolerance,
    }
    summary["gate"] = {
        "enforced": args.enforce,
        "final_tolerance": args.tolerance,
        "max_intermediate_tolerance": args.max_intermediate_tolerance,
        "speed_tolerance_mps": args.speed_tolerance,
        "position_tolerance_m": args.position_tolerance,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.enforce and not summary["gate"]["passed"]:
        print("Parity enforcement failed: " +
              ", ".join(summary["gate"]["failed_checks"]), file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, choices=(50, 1000, 8000), default=50,
                        help="Use 50 for smoke, 1000 or 8000 for acceptance runs")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Sample interval; use 1 for first-divergence gating")
    parser.add_argument("--port", type=int, default=6977)
    parser.add_argument("--start-server", action="store_true")
    default_headless = Path(os.environ.get(
        "GAMA_HEADLESS", "/Applications/Gama.app/Contents/headless/gama-headless.sh"))
    parser.add_argument("--gama-headless", type=Path, default=default_headless,
                        help="Installed GAMA 2025.6.4 launcher")
    parser.add_argument("--traffic-jar", type=Path, default=default_headless.parents[1] /
                        "Eclipse/plugins/gama.extension.traffic_2025.6.4.jar",
                        help="Installed 2025.6.4 traffic plugin JAR for manifest hashing")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Maximum relative error threshold for enforcement")
    parser.add_argument("--speed-tolerance", type=float, default=0.05,
                        help="Per-vehicle speed tolerance in m/s")
    parser.add_argument("--position-tolerance", type=float, default=0.05,
                        help="Per-vehicle road-position tolerance in m")
    parser.add_argument("--python-mode", choices=("safe", "gama_2025_6_4"),
                        default="gama_2025_6_4",
                        help="Python transition profile; parity runs should use gama_2025_6_4")
    parser.add_argument("--max-intermediate-tolerance", type=float, default=0.2,
                        help="Maximum relative error threshold at any sampled intermediate checkpoint (default: 0.20)")
    parser.add_argument("--enforce", action="store_true",
                        help="Exit non-zero if parity thresholds are violated")
    parser.add_argument("--output", type=Path,
                        default=Path("comparison_results/lvl_python_gama.csv"))
    options = parser.parse_args()
    if options.horizon <= 0 or options.sample_every <= 0:
        parser.error("Horizon and sample interval must be positive")
    if min(options.tolerance, options.max_intermediate_tolerance,
           options.speed_tolerance, options.position_tolerance) < 0:
        parser.error("Tolerances must be nonnegative")
    if options.enforce and options.sample_every != 1:
        parser.error("--enforce requires --sample-every 1 to guarantee first divergence")
    if options.enforce and not options.start_server:
        parser.error("--enforce requires --start-server to use the configured installed runtime")
    if "2025.6.4" not in options.traffic_jar.name:
        parser.error("--traffic-jar must identify the installed GAMA 2025.6.4 traffic plugin")
    if not options.gama_headless.is_file() or not options.traffic_jar.is_file():
        parser.error("Installed GAMA 2025.6.4 launcher and traffic JAR must both exist")
    if not asyncio.run(main(options)):
        raise SystemExit(1)
