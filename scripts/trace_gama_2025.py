#!/usr/bin/env python3
"""Write a bounded, paired per-tick Le Van Luong GAMA/Python trace.

The script intentionally observes the installed GAMA runtime through the
existing headless server API. It does not modify the model or either simulator.
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path[:0] = [str(ROOT), str(ROOT / "python-simulation" / "simulation"), str(ROOT / "train" / "logic")]

from train.logic.gama_env import GamaTrafficEnv
from train.logic.server_manager import GamaServerManager
from python_simulation.demand import generate_demand, demand_parameters
from python_simulation.python_sim import ARMS, DT, LeVanLuongSimulator, road_lengths

MODEL = ROOT / "gama-simulation/simulation/models/LeVanLuong.gaml"
GAMA_TAG = "2025.06.4"
GAMA_COMMIT = "7cde6fac892138a161d75c7d02d112a981cbfcb6"
SPEED_TOLERANCE_MPS = 0.05
POSITION_TOLERANCE_M = 0.05
VEHICLES_EXPR = ('"" + (vehicle collect ("" + each.demand_row_id + ":" + '
                 'each.speed + ":" + each.lowest_lane + ":" + '
                 'each.num_lanes_occupied + ":" + '
                 '(each.current_road = nil ? "nil" : road(each.current_road).rid) + ":" + '
                 'each.distance_to_current_target + ":" + each.time_since_lane_change + ":" + '
                 'each.leading_distance + ":" + each.lane_change_cooldown + ":" + '
                 '(each.next_road = nil ? "nil" : road(each.next_road).rid)))')
STATE_EXPR = ('"" + cycle + "|" + phase + "|" + total_vehicles_created + "|" + '
              'total_vehicles_passed + "|" + queue_time_integral + "|" + '
              'system_time_integral + "|" + weighted_delay_integral + "|" + green_ticks')
VEHICLE_RE = re.compile(
    r"(-?\d+):([\d.Ee+-]+):(-?\d+):(\d+):([A-Za-z0-9_]+|nil):"
    r"([\d.Ee+-]+):([\d.Ee+-]+):(?:([\d.Ee+-]+)|Infinity|NaN):([\d.Ee+-]+):"
    r"([A-Za-z0-9_]+|nil)"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tagged_source_sha(path):
    result = subprocess.run(["git", "-C", str(ROOT / "gama"), "show",
                             f"{GAMA_TAG}:{path}"], check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def parse_gama(raw, roads):
    values = {}
    for match in VEHICLE_RE.finditer(raw):
        (row_id, speed, lane, occupied, road, remaining, since_lane_change,
         leader_gap, cooldown, next_road) = match.groups()
        road = None if road == "nil" else road
        next_road = None if next_road == "nil" else next_road
        values[int(row_id)] = {
            "road": road, "road_state": "pending_entry" if road is None else "on_road",
            "lane": int(lane), "speed_mps": float(speed),
            "occupied": int(occupied), "remaining_m": float(remaining),
            "position_on_road_m": (roads[road]["length"] - float(remaining)
                                   if road in roads else None),
            "time_since_lane_change": float(since_lane_change),
            "leader_gap": float(leader_gap) if leader_gap is not None else math.inf,
            "lane_change_cooldown": float(cooldown),
            "next_road": next_road,
        }
    return values


def serialize_python_vehicle(sim, vehicle):
    """Match comparison-gate road semantics, preserving GAMA's pending entry."""
    pending = sim.mode == "gama_2025_6_4" and not vehicle.gama_entered
    if pending:
        road, position, next_road, road_state = None, None, f"{vehicle.arm}_in", "pending_entry"
    else:
        road, position = sim._road_state(vehicle)
        next_road = (f"{vehicle.arm}_{vehicle.to_arm}"
                     if road == f"{vehicle.arm}_in" else
                     f"{vehicle.to_arm}_out"
                     if road == f"{vehicle.arm}_{vehicle.to_arm}" else None)
        road_state = "on_road"
    return {"road": road, "road_state": road_state, "gama_entered": vehicle.gama_entered,
            "lane": vehicle.lane, "speed_mps": vehicle.speed,
            "position_on_road_m": position,
            "time_since_lane_change": vehicle.time_since_lane_change,
            "occupied": vehicle.occupied,
            "leader_gap": (vehicle.leading_distance
                           if math.isfinite(vehicle.leading_distance) else None),
            "lane_change_cooldown": 4.0, "next_road": next_road}


async def run(args):
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    demand_path = out.with_name(out.stem + "_demand.csv")
    demand_hash = generate_demand(demand_path, args.seed, args.ticks)
    demand_rows = demand_path.read_text(encoding="utf-8").splitlines()
    demand_by_cycle = {}
    demand_meta = {}
    for row_id, line in enumerate(demand_rows):
        fields = line.split(",")
        cycle = int(fields[0])
        demand_by_cycle.setdefault(cycle, []).append(row_id)
        demand_meta[row_id] = {"requested_lane": int(fields[5]),
                               "requested_speed_kmh": float(fields[3]),
                               "arm_index": int(fields[1]), "is_car": bool(int(fields[2])),
                               "turn": int(fields[4])}

    roads = road_lengths()
    python = LeVanLuongSimulator(seed=args.seed, demand_file=demand_path,
                                 mode=args.python_mode)
    manager = GamaServerManager(
        port=args.port, workdir=str(out.parent / "gama_2025_workspace"),
        log_file=str(out.with_suffix(".gama.log")), startup_timeout=30.0)
    env = None
    first_structural_divergence = None
    first_numeric_divergence = None
    try:
        manager.start(reuse_running=False)
        env = GamaTrafficEnv(port=args.port, model_path=str(MODEL),
                             experiment="python_parity", verbose=False)
        await env.setup()
        await env.load_experiment(parameters=[
            *demand_parameters(demand_path),
            {"name": "traffic_seed", "type": "float", "value": str(float(args.seed))}])
        with out.open("w", encoding="utf-8") as stream:
            for tick in range(args.ticks + 1):
                raw_state = str(await env._exec_expression(STATE_EXPR)).split("|")
                if len(raw_state) != 8:
                    raise RuntimeError(f"Bad GAMA state at tick {tick}: {raw_state}")
                gs = {"cycle": int(float(raw_state[0])), "phase": raw_state[1],
                      "created": int(float(raw_state[2])), "passed": int(float(raw_state[3])),
                      "queue_time_integral": float(raw_state[4]),
                      "system_time_integral": float(raw_state[5]),
                      "weighted_delay_integral": float(raw_state[6]),
                      "green_ticks": int(float(raw_state[7]))}
                ga = parse_gama(str(await env._exec_expression(VEHICLES_EXPR)), roads)
                py = {v.demand_row_id: v for v in python.vehicles}
                py_state = python.snapshot()
                if tick == 0:
                    previous_gs, previous_py_state = gs, py_state
                all_ids = sorted(set(ga) | set(py))
                for rid in all_ids:
                    gv, pv = ga.get(rid), py.get(rid)
                    if gv is None or pv is None:
                        mismatch = "presence"
                    else:
                        py_vehicle = serialize_python_vehicle(python, pv)
                        py_road = py_vehicle["road"]
                        if gv["road"] is None and py_road is None:
                            mismatch = "lane" if gv["lane"] != pv.lane else None
                        elif gv["road"] is None or py_road is None:
                            mismatch = "pending_road"
                        elif gv["road"] != py_road:
                            mismatch = "road"
                        elif gv["lane"] != pv.lane:
                            mismatch = "lane"
                        else:
                            mismatch = None
                    numeric = None
                    if gv is not None and pv is not None:
                        speed_delta = abs(gv["speed_mps"] - py_vehicle["speed_mps"])
                        position_delta = (None if gv["position_on_road_m"] is None else
                                          abs(gv["position_on_road_m"] -
                                              py_vehicle["position_on_road_m"]))
                        if (speed_delta > SPEED_TOLERANCE_MPS or
                                (position_delta is not None and
                                 position_delta > POSITION_TOLERANCE_M)):
                            numeric = {"speed_delta_mps": speed_delta,
                                       "position_delta_m": position_delta}
                            if first_numeric_divergence is None:
                                first_numeric_divergence = {
                                    "tick": tick, "demand_row_id": rid, **numeric}
                    if mismatch and first_structural_divergence is None:
                        first_structural_divergence = {"tick": tick,
                                                       "demand_row_id": rid,
                                                       "kind": mismatch}
                    record = {
                        "tick": tick, "demand_row_id": rid,
                        "demand_spawn_cycle": next((c for c, ids in demand_by_cycle.items()
                                                     if rid in ids), None),
                        "demand": demand_meta.get(rid),
                        "gama": gv, "python": py_vehicle if gv is not None and pv is not None else None,
                        "gama_phase": gs["phase"], "python_phase": py_state["phase"],
                        "gama_created": gs["created"], "python_created": py_state["created"],
                        "gama_passed": gs["passed"], "python_passed": py_state["passed"],
                        "tick_event": {"spawned_demand_rows": demand_by_cycle.get(tick, []),
                                       "gama_integral_delta": {
                                           key: gs[key] - previous_gs[key] for key in
                                           ("queue_time_integral", "system_time_integral",
                                            "weighted_delay_integral")},
                                       "python_integral_delta": {
                                           key: py_state[key] - previous_py_state[key] for key in
                                           ("queue_time_integral", "system_time_integral",
                                            "weighted_delay_integral")}},
                        "divergence": mismatch,
                        "numeric_divergence": numeric,
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                if tick < args.ticks:
                    python.step(1)
                    await env.step_cycles(1)
                    previous_gs, previous_py_state = gs, py_state
        manifest = {
            "runtime_target": {"version": GAMA_TAG, "tag_commit": GAMA_COMMIT,
                               "installed_headless": str(args.headless)},
            "seed": args.seed, "ticks": args.ticks, "step_seconds": DT,
            "python_mode": args.python_mode,
            "trace": str(out), "demand_csv": str(demand_path),
            "first_structural_divergence": first_structural_divergence,
            "first_numeric_divergence": first_numeric_divergence,
            "gama_tag_source_commit": GAMA_COMMIT,
            "sha256": {"gaml_model": sha(MODEL),
                       "python_sim": sha(ROOT / "python-simulation/simulation/python_simulation/python_sim.py"),
                       "demand_generator": sha(ROOT / "python-simulation/simulation/python_simulation/demand.py"),
                       "demand_csv": demand_hash,
                       "gama_2025_driving_skill": tagged_source_sha(
                           "gama.extension.traffic/src/gama/extension/traffic/driving/DrivingSkill.java"),
                       "gama_2025_mobil": tagged_source_sha(
                           "gama.extension.traffic/src/gama/extension/traffic/driving/carfollowing/MOBIL.java"),
                       "headless_launcher": sha(args.headless),
                       "traffic_plugin_2025_6_4": sha(args.traffic_jar)},
            "command": [sys.executable, str(Path(__file__).resolve()), "--seed", str(args.seed),
                        "--ticks", str(args.ticks), "--port", str(args.port),
                        "--python-mode", args.python_mode,
                        "--output", str(out)],
            "semantics": {"GAMA position": "road length minus distance_to_current_target",
                          "Python position": "LeVanLuongSimulator._road_state progress",
                          "numeric_tolerances": {"speed_mps": SPEED_TOLERANCE_MPS,
                                                 "position_m": POSITION_TOLERANCE_M},
                          "measurement": "per-tick integral differences observed after completed tick; model measures before vehicle move"},
        }
        out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    finally:
        manager.stop(force=True)
        if env is not None:
            try:
                env.disconnect()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=50,
                        help="Bounded capture length (1..70 ticks)")
    parser.add_argument("--python-mode", choices=("safe", "gama_2025_6_4"),
                        default="safe", help="Python simulator mode; default safe")
    parser.add_argument("--port", type=int, default=6994)
    parser.add_argument("--output", type=Path,
                        default=Path("audits/gama_source_20261112/gama_2025_seed1_50.jsonl"))
    parser.add_argument("--headless", type=Path, default=Path(
        "/Applications/Gama.app/Contents/headless/gama-headless.sh"))
    parser.add_argument("--traffic-jar", type=Path, default=Path(
        "/Applications/Gama.app/Contents/Eclipse/plugins/gama.extension.traffic_2025.6.4.jar"))
    args = parser.parse_args()
    if not 1 <= args.ticks <= 70:
        parser.error("Trace capture is bounded to 1..70 ticks")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
