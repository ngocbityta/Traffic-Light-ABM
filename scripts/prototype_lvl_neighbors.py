#!/usr/bin/env python3
"""Benchmark a parity-preserving runtime prototype for Python LVL neighbors.

The prototype only monkeypatches LeVanLuongSimulator._neighbors in this
process. It local-binds repeated attribute/module lookups while keeping the
baseline algorithm, comparisons, and iteration order unchanged.
"""
import argparse
import bisect
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LeVanLuongIntersection.python_sim import LeVanLuongSimulator


def neighbors_locals(self, vehicle, lane, lane_maps):
    road, position = self._neighbor_road_state[id(vehicle)]
    leader = follower = None
    front = back = math.inf
    front_distance = back_distance = math.inf
    lane_entries = lane_maps[road]
    occupied = vehicle.occupied
    vehicle_length = vehicle.length
    bisect_left = bisect.bisect_left
    for sublane in range(lane, lane + occupied):
        entries = lane_entries[sublane]
        index = bisect_left(entries, (position, -1))
        ahead = index
        while ahead < len(entries) and entries[ahead][2] is vehicle:
            ahead += 1
        if ahead < len(entries):
            other_position, _, other = entries[ahead]
            distance = other_position - position
            if distance < front_distance:
                front_distance = distance
                leader = other
                front = distance - (vehicle_length + other.length) / 2
        behind = index - 1
        while behind >= 0 and entries[behind][2] is vehicle:
            behind -= 1
        if behind >= 0:
            other_position, _, other = entries[behind]
            distance = position - other_position
            if distance < back_distance:
                back_distance = distance
                follower = other
                back = distance - (vehicle_length + other.length) / 2
    if leader is None:
        next_road = None
        arm = vehicle.arm
        to_arm = vehicle.to_arm
        if road == f"{arm}_in":
            green = (arm in ("NE", "SW") and self.phase == "LVL") or (
                arm in ("SE", "NW") and self.phase == "HDT")
            if green:
                next_road = f"{arm}_{to_arm}"
        elif road == f"{arm}_{to_arm}":
            next_road = f"{to_arm}_out"
        if next_road in lane_maps:
            next_lanes = self.roads[next_road]["num_lanes"]
            next_lane = min(lane, next_lanes - occupied)
            road_length = self.roads[road]["length"]
            for sublane in range(next_lane, next_lane + occupied):
                entries = lane_maps[next_road][sublane]
                if entries:
                    other_position, _, other = entries[0]
                    distance = road_length - position + other_position
                    if distance < front_distance:
                        front_distance = distance
                        leader = other
                        front = distance - (vehicle_length + other.length) / 2
    return leader, front, follower, back


def vehicle_trace(sim):
    return [(v.demand_row_id, v.arm, v.to_arm, v.maneuver, v.lane,
             v.occupied, v.position, v.speed, v.time_since_lane_change)
            for v in sim.vehicles]


def run(demand, seed, horizon, optimized, trace_ticks):
    if optimized:
        LeVanLuongSimulator._neighbors = neighbors_locals
    else:
        LeVanLuongSimulator._neighbors = BASELINE_NEIGHBORS
    sim = LeVanLuongSimulator(seed=seed, demand_file=demand, train_mode=False)
    snapshots, traces = [], []
    started = time.perf_counter()
    for tick in range(1, horizon + 1):
        sim.step(1)
        if tick <= trace_ticks:
            traces.append((tick, vehicle_trace(sim)))
        if tick % 400 == 0 or tick == horizon:
            snapshots.append(sim.snapshot())
    elapsed = time.perf_counter() - started
    return {"elapsed_sec": elapsed, "ticks_per_sec": horizon / elapsed,
            "snapshots": snapshots, "traces": traces}


BASELINE_NEIGHBORS = LeVanLuongSimulator._neighbors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demand", type=Path, default=ROOT / "reports/python_parity_20261110/lvl_queue_crossroad_8000_seed1_demand.csv")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=8000)
    p.add_argument("--trace-ticks", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if not a.demand.is_file():
        p.error(f"demand file not found: {a.demand}")
    if a.horizon <= 0 or a.horizon % 400 or not 0 <= a.trace_ticks <= a.horizon or a.repeats < 1:
        p.error("horizon must be positive/multiple of 400; trace ticks within horizon; repeats positive")
    runs = {"baseline": [], "prototype": []}
    parity = None
    # Alternate order by repeat to reduce systematic thermal/order bias.
    for rep in range(a.repeats):
        modes = ("baseline", "prototype") if rep % 2 == 0 else ("prototype", "baseline")
        for mode in modes:
            result = run(a.demand, a.seed, a.horizon, mode == "prototype", a.trace_ticks)
            runs[mode].append({"elapsed_sec": result["elapsed_sec"],
                               "ticks_per_sec": result["ticks_per_sec"]})
            if parity is None:
                parity = {mode: result}
            elif mode not in parity:
                parity[mode] = result
    before, after = parity["baseline"], parity["prototype"]
    snapshot_equal = before["snapshots"] == after["snapshots"]
    trace_equal = before["traces"] == after["traces"]
    result = {
        "python": sys.version.split()[0], "seed": a.seed, "horizon": a.horizon,
        "demand": str(a.demand.resolve()), "demand_bytes": a.demand.stat().st_size,
        "demand_sha256": hashlib.sha256(a.demand.read_bytes()).hexdigest(),
        "trace_ticks": a.trace_ticks,
        "parity": {"snapshots_every_400_equal": snapshot_equal,
                   "vehicle_trace_equal": trace_equal,
                   "trace_record_count": len(before["traces"])},
        "runs": runs,
        "median_seconds": {k: statistics.median(v["elapsed_sec"] for v in values)
                           for k, values in runs.items()},
        "median_ticks_per_sec": {k: statistics.median(v["ticks_per_sec"] for v in values)
                                  for k, values in runs.items()},
    }
    b = result["median_seconds"]["baseline"]
    q = result["median_seconds"]["prototype"]
    result["speedup"] = b / q
    result["percent_change_elapsed"] = (b / q - 1) * 100
    if not snapshot_equal or not trace_equal:
        raise AssertionError("Prototype changed 400-tick snapshots or first-200-tick vehicle trace")
    rendered = json.dumps(result, indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
