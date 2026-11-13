#!/usr/bin/env python3
"""Summarize paired per-vehicle GAMA/Python LVL traces.

Example: python scripts/analyze_lvl_vehicle_parity.py TRACE --seed 1 --horizon 1000
"""
import argparse
import csv
import hashlib
import json
import statistics
import tempfile
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON_SIM_ROOT = ROOT / "python-simulation" / "simulation"
sys.path.insert(0, str(PYTHON_SIM_ROOT))
from python_simulation.demand import generate_demand
from python_simulation.python_sim import LeVanLuongSimulator, road_lengths


def summarize(trace, seed, horizon, replay_current_python=False):
    with tempfile.TemporaryDirectory(prefix="lvl_analyze_") as d:
        demand_path = Path(d) / "demand.csv"
        digest = generate_demand(demand_path, seed, horizon)
        demand = list(csv.reader(demand_path.open()))
    roads = road_lengths()
    def road_position(row_id, road, remaining):
        if not 0 <= row_id < len(demand):
            raise ValueError(f"Trace row_id {row_id} has no matching demand row (demand rows: {len(demand)})")
        cycle, arm_index, car, speed, turn, lane = demand[row_id]
        arm = ("NE", "SE", "SW", "NW")[int(arm_index)]
        # Connector IDs and their destination are defined by the current road map.
        turn_name = ("left", "right", "through")[int(turn)]
        conn = next(r for r in roads.values()
                    if r["kind"] == "connector" and r["arm"] == arm and
                    r["turn"] == turn_name)
        incoming = roads[f"{arm}_in"]["length"]
        if road == f"{arm}_in":
            return incoming - remaining
        if road == conn["id"]:
            return incoming + conn["length"] - remaining
        if road == f"{conn['to_arm']}_out":
            return incoming + conn["length"] + roads[road]["length"] - remaining
        raise ValueError(f"Trace route mismatch for row_id {row_id}: road {road!r} is not on expected route "
                         f"{arm}_in -> {conn['id']} -> {conn['to_arm']}_out")

    rows = [json.loads(line) for line in Path(trace).read_text().splitlines() if line.strip()]
    python_source = "replayed" if replay_current_python else "embedded"
    simulator_sha256 = hashlib.sha256((ROOT / "python-simulation/simulation/python_simulation/python_sim.py").read_bytes()).hexdigest()
    model_sha256 = hashlib.sha256((ROOT / "gama-simulation/simulation/models/LeVanLuong.gaml").read_bytes()).hexdigest()
    demand_source_sha256 = hashlib.sha256((ROOT / "python-simulation/simulation/python_simulation/demand.py").read_bytes()).hexdigest()
    replay_dir = tempfile.TemporaryDirectory(prefix="lvl_replay_") if replay_current_python else None
    replay_demand = None
    if replay_dir is not None:
        replay_demand = Path(replay_dir.name) / "demand.csv"
        generate_demand(replay_demand, seed, horizon)
    replay_sim = LeVanLuongSimulator(seed=seed, demand_file=replay_demand) if replay_current_python else None
    replay_tick = 0
    by_tick = defaultdict(list)
    by_road = defaultdict(list)
    by_kind = defaultdict(list)
    first_by_vehicle = {}
    for x in rows:
        g, rid = x.get("gama"), int(x["row_id"])
        if not 0 <= rid < len(demand):
            raise ValueError(f"Trace row_id {rid} at tick {x.get('tick')} has no matching demand row "
                             f"(demand rows: {len(demand)})")
        if replay_sim is not None:
            tick = int(x["tick"])
            if tick < replay_tick:
                raise ValueError("Trace rows must be ordered by nondecreasing tick for replay")
            if tick > replay_tick:
                replay_sim.step(tick - replay_tick)
                replay_tick = tick
            current = next((v for v in replay_sim.vehicles if v.demand_row_id == rid), None)
            x = dict(x, python_position=current.position if current else None,
                     python_speed=current.speed if current else None,
                     python_lane=current.lane if current else None)
        if not g or x.get("python_lane") is None:
            continue
        pos = (None if g["road"] in (None, "nil") else
               road_position(rid, g["road"], g["remaining"]))
        dp = (None if pos is None or x.get("python_position") is None else
              float(x["python_position"]) - pos)
        ds = float(x["python_speed"]) - float(g["speed"])
        lane_bad = int(x["python_lane"] != g["lane"])
        road_bad = int(x.get("python_road") is not None and x["python_road"] != g["road"])
        rec = (abs(dp) if dp is not None else None, abs(ds), lane_bad, road_bad)
        by_tick[x["tick"]].append((rid, rec, dp, ds, g["road"], "car" if int(demand[rid][2]) else "motorbike"))
        by_road[g["road"]].append(rec)
        kind = "car" if int(demand[rid][2]) else "motorbike"
        by_kind[kind].append(rec)
        if lane_bad and rid not in first_by_vehicle:
            first_by_vehicle[rid] = {"tick": x["tick"], "road": g["road"], "type": kind}

    def agg(vals):
        if not vals:
            return {"n": 0, "lane_mismatch_rate": None, "road_mismatch_rate": None,
                    "speed_mae_mps": None, "position_mae_m": None,
                    "position_sample_n": 0}
        return {"n": len(vals), "lane_mismatch_rate": sum(v[2] for v in vals)/len(vals),
                "road_mismatch_rate": sum(v[3] for v in vals)/len(vals),
                "speed_mae_mps": statistics.fmean(v[1] for v in vals),
                "position_mae_m": statistics.fmean(v[0] for v in vals if v[0] is not None)
                    if any(v[0] is not None for v in vals) else None,
                "position_sample_n": sum(v[0] is not None for v in vals)}
    allvals = [v[1] for bucket in by_tick.values() for v in bucket]
    first_tick = next((t for t in sorted(by_tick) if any(v[1][2] for v in by_tick[t])), None)
    first = None if first_tick is None else next({"tick": first_tick, "row_id": v[0], "road": v[4], "type": v[5]}
                                                  for v in by_tick[first_tick] if v[1][2])
    result = {"trace": str(trace), "seed": seed, "horizon": horizon, "demand_sha256": digest,
            "python_state_source": python_source,
            "python_simulator_sha256": simulator_sha256,
            "gama_model_sha256": model_sha256,
            "demand_source_sha256": demand_source_sha256,
            "sample_ticks": sorted(by_tick),
            "position_method": "GAMA route offset + current-road length - distance_to_current_target; Python cumulative route position",
            "sample_count": len(allvals), "first_lane_mismatch": first,
            "vehicles_with_lane_mismatch": len(first_by_vehicle),
            "first_lane_mismatch_by_vehicle": first_by_vehicle,
            "overall": agg(allvals),
            "by_milestone": {str(tick): agg([record[1] for record in bucket])
                             for tick, bucket in sorted(by_tick.items())},
            "by_road": {k: agg(v) for k, v in sorted(by_road.items())},
            "by_vehicle_type": {k: agg(v) for k, v in sorted(by_kind.items())}}
    if replay_dir is not None:
        replay_dir.cleanup()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--horizon", type=int, required=True, help="Demand horizon used to create the trace")
    p.add_argument("--replay-current-python", action="store_true",
                   help="Replay current Python simulator from seed/demand instead of using embedded python_* trace fields")
    p.add_argument("--output", type=Path, help="Optional JSON report path")
    a = p.parse_args()
    result = summarize(a.trace, a.seed, a.horizon, a.replay_current_python)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(encoded + "\n")
    print(encoded)

if __name__ == "__main__":
    main()
