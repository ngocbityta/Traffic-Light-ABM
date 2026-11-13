#!/usr/bin/env python3
"""Measure link-road safety invariants for the current LVL Python simulator.

Connector interactions inside the intersection are intentionally excluded.
"""

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON_SIM_ROOT = ROOT / "python-simulation" / "simulation"
sys.path.insert(0, str(PYTHON_SIM_ROOT))

from python_simulation.python_sim import LeVanLuongSimulator, STOP_MARGIN
from python_simulation.demand import generate_demand


def audit(seed, horizon, flow, paired_demand=False, train_mode=False):
    if paired_demand:
        with tempfile.TemporaryDirectory(prefix="lvl_safety_demand_") as directory:
            demand = Path(directory) / "demand.csv"
            demand_sha256 = generate_demand(
                demand, seed=seed, horizon=horizon, flow_per_approach=flow)
            sim = LeVanLuongSimulator(seed=seed, demand_file=demand,
                                     train_mode=train_mode)
    else:
        demand_sha256 = None
        sim = LeVanLuongSimulator(seed=seed, flow_per_approach=flow,
                                 train_mode=train_mode)
    result = {
        "seed": seed, "horizon": horizon, "flow_per_approach": flow,
        "demand_sha256": demand_sha256,
        "controller": "alternating_train_actions" if train_mode else "fixed_time",
        "road_clearance_violations": 0,
        "red_stop_violations": 0,
        "red_clearance_vehicle_ticks": 0,
        "stopped_lateral_moves": 0,
        "turn_corridor_violations": 0,
        "negative_exit_progress": 0,
        "backward_vehicle_ticks": 0,
        "max_backward_metres": 0.0,
        "nonfinite_states": 0,
    }
    committed_on_green = set()
    for _ in range(horizon):
        before = {id(v): (*sim._road_state(v), v.current_lateral_shift, v.speed)
                  for v in sim.vehicles}
        committed_on_green.intersection_update(before)
        if train_mode and sim.dqn_waiting_action:
            sim.set_action(2 if sim.phase == "LVL" else 0)
        sim.step()
        by_road = {}
        for v in sim.vehicles:
            road, progress = sim._road_state(v)
            if not all(map(math.isfinite, (v.position, v.speed, v.current_lateral_shift))):
                result["nonfinite_states"] += 1
                continue
            if not road.endswith(("_in", "_out")):
                continue
            by_road.setdefault(road, []).append((progress, v))
            previous = before.get(id(v))
            if previous and previous[0] == road:
                backward = previous[1] - progress
                if backward > 1e-8:
                    result["backward_vehicle_ticks"] += 1
                    result["max_backward_metres"] = max(
                        result["max_backward_metres"], backward)
                if (previous[3] < 0.1 and v.speed < 0.1 and
                        abs(previous[2] - v.current_lateral_shift) > 1e-8):
                    result["stopped_lateral_moves"] += 1
            if road.endswith("_out"):
                if progress < -1e-8:
                    result["negative_exit_progress"] += 1
                continue
            green = ((v.arm in ("NE", "SW") and sim.phase == "LVL") or
                     (v.arm in ("SE", "NW") and sim.phase == "HDT"))
            front_over_line = (v.position + v.length / 2 + STOP_MARGIN >
                               v.approach_end + 1e-8)
            if green and front_over_line:
                committed_on_green.add(id(v))
            elif not green and front_over_line:
                if id(v) in committed_on_green and v.speed >= 0.1:
                    result["red_clearance_vehicle_ticks"] += 1
                else:
                    result["red_stop_violations"] += 1
            if v.maneuver != "through" and 0 <= sim._stop_position(v) - v.position <= 10:
                if sim._route_target_lane(v) != v.lane:
                    result["turn_corridor_violations"] += 1
        for entries in by_road.values():
            entries.sort(key=lambda item: item[0])
            for i, (p, v) in enumerate(entries):
                for q, other in entries[i + 1:]:
                    if q - p > 8:
                        break
                    lateral = abs(v.current_lateral_shift - other.current_lateral_shift)
                    required_lateral = sim._half_width(v) + sim._half_width(other) + 0.1
                    if lateral >= required_lateral:
                        continue
                    gap = q - p - (v.length + other.length) / 2
                    if gap + 1e-8 < max(1.0, v.gap, other.gap):
                        result["road_clearance_violations"] += 1
    result.update(created=sim.created, passed=sim.passed, active=len(sim.vehicles),
                  road_safety_corrections=sim.road_safety_corrections)
    result["max_backward_metres"] = round(result["max_backward_metres"], 6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,2,3", help="Comma-separated integer seeds")
    parser.add_argument("--horizon", type=int, default=8000)
    parser.add_argument("--flow", type=float, default=2400)
    parser.add_argument("--paired-demand", action="store_true",
                        help="Generate and replay the same CSV demand used for paired GAMA runs")
    parser.add_argument("--train-mode", action="store_true",
                        help="Alternate LVL/HDT actions at every training decision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if args.horizon < 0:
        parser.error("horizon must be nonnegative")
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    if not seeds:
        parser.error("at least one seed is required")
    rows = [audit(seed, args.horizon, args.flow, args.paired_demand,
                  args.train_mode) for seed in seeds]
    source = ROOT / "python-simulation" / "simulation" / "python_simulation" / "python_sim.py"
    roads = ROOT / "python-simulation" / "simulation" / "python_simulation" / "includes" / "roads.geojson"
    report = {"scope": "incoming and outgoing roads; connector collisions deferred",
              "python_sim_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "roads_sha256": hashlib.sha256(roads.read_bytes()).hexdigest(),
              "results": rows}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.enforce and any(any(row[key] for key in (
            "road_clearance_violations", "red_stop_violations",
            "stopped_lateral_moves", "turn_corridor_violations",
            "negative_exit_progress", "nonfinite_states")) for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
