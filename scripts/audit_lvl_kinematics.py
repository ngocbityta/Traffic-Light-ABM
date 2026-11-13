#!/usr/bin/env python3
"""Read-only kinematic audit for the Le Van Luong Python simulator.

Tracks world-center motion, speed consistency, polyline heading changes, and
road-seam continuity. The simulator is observed but never modified by this
script; it does not change connector arbitration or geometry.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "python-simulation" / "simulation" / "python_simulation"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python-simulation" / "simulation"))

PAIRED_SEED1 = REPO / "audits/gama_source_20261112/parity_seed1_1000_v4_demand.csv"
DT = 0.2


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def stop_kinematics(displacement_m, dt, start_speed, end_speed, decel_limit):
    """Return observed speed mismatch and physical deceleration diagnostics."""
    observed_speed = displacement_m / dt if dt > 0 else math.inf
    mean_speed = (start_speed + end_speed) / 2
    deceleration = max(0.0, (start_speed - end_speed) / dt) if dt > 0 else math.inf
    return {
        "observed_speed_mps": observed_speed,
        "end_speed_mismatch_mps": abs(observed_speed - end_speed),
        "mean_speed_mismatch_mps": abs(observed_speed - mean_speed),
        "deceleration_mps2": deceleration,
        "exceeds_decel_limit": deceleration > decel_limit + 1e-9,
    }


def seam_center_jump(old_road, new_road, seam_progress, new_shift, old_shift):
    """Compare exact old endpoint with the new-road pose chosen for that seam."""
    before = old_road.pose(old_road.length, old_shift)
    after = new_road.pose(seam_progress, new_shift)
    return math.hypot(after.x - before.x, after.y - before.y)


def _percentiles(values, quantiles=(0.5, 0.9, 0.95, 0.99, 1.0)):
    values = sorted(values)
    if not values:
        return {str(q): None for q in quantiles}
    return {str(q): values[min(len(values) - 1, int((len(values) - 1) * q))]
            for q in quantiles}


def _road_pose(sim, vehicle):
    road_id, progress = sim._road_state(vehicle)
    pose = sim._geometry_roads[road_id].pose(progress, vehicle.current_lateral_shift)
    return road_id, progress, pose


def _segment_index(road, progress):
    return next((i for i in range(len(road.cumulative) - 1)
                 if progress < road.cumulative[i + 1]), len(road.cumulative) - 2)


def _vehicle_key(vehicle):
    return ("demand", vehicle.demand_row_id) if vehicle.demand_row_id >= 0 else ("object", id(vehicle))


def audit(seed=1, ticks=400, mode="safe", demand_file=None, dt=DT):
    """Run a bounded replay and return JSON-serializable kinematic metrics."""
    from python_simulation.python_sim import LeVanLuongSimulator

    demand_file = Path(demand_file) if demand_file else None
    paths = {
        "python_sim.py": ROOT / "python_sim.py",
        "vehicle_geometry.py": ROOT / "vehicle_geometry.py",
        "junction_conflicts.py": ROOT / "junction_conflicts.py",
        "safe_startup.py": ROOT / "safe_startup.py",
        "gama_2025_idm_mobil.py": ROOT / "gama_2025_idm_mobil.py",
        "gama_2025_transition.py": ROOT / "gama_2025_transition.py",
        "roads.geojson": ROOT / "includes/roads.geojson",
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    demand_hash = sha256(demand_file) if demand_file else None
    sim = LeVanLuongSimulator(seed=seed, mode=mode, demand_file=demand_file)
    distances, end_mismatches, mean_mismatches, decelerations = [], [], [], []
    heading_jumps, seam_jumps, vertex_jumps, hard_stops, stopped_moves = [], [], [], [], []
    observed_steps = 0

    for _ in range(ticks):
        before = {}
        for vehicle in sim.vehicles:
            road_id, progress, pose = _road_pose(sim, vehicle)
            before[_vehicle_key(vehicle)] = {
                "vehicle": vehicle, "road": road_id, "progress": progress,
                "pose": pose, "speed": vehicle.speed,
                "shift": vehicle.current_lateral_shift,
            }
        sim.step()
        for vehicle in sim.vehicles:
            old = before.get(_vehicle_key(vehicle))
            if old is None:
                continue
            road_id, progress, pose = _road_pose(sim, vehicle)
            displacement = math.hypot(pose.x - old["pose"].x, pose.y - old["pose"].y)
            kin = stop_kinematics(displacement, dt, old["speed"], vehicle.speed,
                                  vehicle.decel)
            observed_steps += 1
            distances.append(displacement)
            end_mismatches.append(kin["end_speed_mismatch_mps"])
            mean_mismatches.append(kin["mean_speed_mismatch_mps"])
            decelerations.append(kin["deceleration_mps2"])
            angle = abs(wrap_angle(pose.heading - old["pose"].heading))
            event = {
                "tick": sim.cycle, "vehicle_id": str(_vehicle_key(vehicle)[1]),
                "from_road": old["road"], "to_road": road_id,
                "from_progress_m": old["progress"], "to_progress_m": progress,
                "displacement_m": displacement, "heading_jump_deg": math.degrees(angle),
                "start_speed_mps": old["speed"], "end_speed_mps": vehicle.speed,
            }
            if angle > math.radians(10):
                heading_jumps.append(event)
            if old["road"] != road_id:
                jump = seam_center_jump(
                    sim._geometry_roads[old["road"]], sim._geometry_roads[road_id],
                    vehicle.seam_progress, vehicle.seam_shift, old["shift"])
                seam_jumps.append({**event, "seam_center_jump_m": jump})
            elif _segment_index(sim._geometry_roads[road_id], old["progress"]) != _segment_index(
                    sim._geometry_roads[road_id], progress):
                vertex_jumps.append(event)
            if kin["exceeds_decel_limit"]:
                hard_stops.append({**event, **kin, "decel_limit_mps2": vehicle.decel})
            if old["speed"] < 0.05 and vehicle.speed < 0.05 and displacement > 0.05:
                stopped_moves.append(event)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    stale = hashes_before != hashes_after
    return {
        "schema_version": 1,
        "scope": "safe-mode world kinematics; no simulator or geometry changes",
        "seed": seed, "ticks": ticks, "dt_s": dt, "mode": mode,
        "demand_file": str(demand_file) if demand_file else None,
        "demand_sha256": demand_hash,
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_changed_during_run": stale,
        "sample_stale": stale,
        "observed_vehicle_steps": observed_steps,
        "world_displacement_m_per_tick": _percentiles(distances),
        "abs_world_speed_minus_reported_end_speed_mps": _percentiles(end_mismatches),
        "abs_world_speed_minus_mean_endpoint_speed_mps": _percentiles(mean_mismatches),
        "deceleration_mps2": _percentiles(decelerations),
        "max_deceleration_excess_events": len(hard_stops),
        "max_deceleration_excess_examples": sorted(
            hard_stops, key=lambda e: e["deceleration_mps2"], reverse=True)[:10],
        "heading_jumps_over_10deg": len(heading_jumps),
        "max_heading_jump_deg": max((e["heading_jump_deg"] for e in heading_jumps), default=0.0),
        "heading_jump_examples": sorted(
            heading_jumps, key=lambda e: e["heading_jump_deg"], reverse=True)[:10],
        "seam_transitions": len(seam_jumps),
        "max_seam_center_jump_m": max(
            (e["seam_center_jump_m"] for e in seam_jumps), default=0.0),
        "seam_examples": sorted(
            seam_jumps, key=lambda e: e["seam_center_jump_m"], reverse=True)[:10],
        "vertex_crossings": len(vertex_jumps),
        "max_vertex_heading_jump_deg": max(
            (e["heading_jump_deg"] for e in vertex_jumps), default=0.0),
        "vertex_heading_examples": sorted(
            vertex_jumps, key=lambda e: e["heading_jump_deg"], reverse=True)[:10],
        "stopped_body_moves_over_5cm": len(stopped_moves),
        "stopped_body_move_examples": stopped_moves[:10],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--mode", choices=("safe", "gama_2025_6_4"), default="safe")
    parser.add_argument("--demand-file", type=Path,
                        help="Replay a saved six-column demand CSV")
    parser.add_argument("--paired-seed1-400", action="store_true",
                        help="Replay the saved paired seed 1 demand for 400 ticks")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ticks < 0:
        parser.error("ticks must be nonnegative")
    if args.paired_seed1_400:
        if args.demand_file:
            parser.error("choose either --demand-file or --paired-seed1-400")
        args.demand_file = PAIRED_SEED1
        args.seed = 1
        args.ticks = 400
    if args.demand_file and not args.demand_file.is_file():
        parser.error(f"demand file not found: {args.demand_file}")
    result = audit(args.seed, args.ticks, args.mode, args.demand_file)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
