#!/usr/bin/env python3
"""Audit safe-mode red-to-green queue discharge from fixed or train action runs."""

import argparse
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON_SIM_ROOT = ROOT / "python-simulation" / "simulation"
sys.path.insert(0, str(PYTHON_SIM_ROOT))

from python_simulation.python_sim import ARMS, DT, LeVanLuongSimulator, STOP_MARGIN
from python_simulation.demand import generate_demand


def green_for(arm, phase):
    return ((arm in ("NE", "SW") and phase == "LVL") or
            (arm in ("SE", "NW") and phase == "HDT"))


class DischargeTracker:
    """Collect onset cohorts and first observed movement times."""

    def __init__(self):
        self.events = []
        self.pending = {}

    def observe(self, tick, old_phase, new_phase, before, after):
        after_by_id = {v["id"]: v for v in after}
        for entry in self.pending.values():
            current = after_by_id.get(entry["id"])
            if entry["release_tick"] is None and current and current["speed"] >= 0.1:
                entry["release_tick"] = tick
                entry["release_time_sec"] = round(tick * DT, 3)
        if new_phase not in ("LVL", "HDT") or new_phase == old_phase:
            return
        cohort = {}
        for vehicle in before:
            road, _ = vehicle["road_state"]
            if (road.endswith("_in") and green_for(vehicle["arm"], new_phase)
                    and vehicle["speed"] < 0.1):
                cohort[vehicle["id"]] = {"id": vehicle["id"], "arm": vehicle["arm"],
                    "lane": vehicle["lane"], "occupied": vehicle["occupied"],
                    "speed_before": vehicle["speed"], "release_tick": None,
                    "release_time_sec": None}
        for ident, entry in cohort.items():
            current = after_by_id.get(ident)
            if current and current["speed"] >= 0.1:
                entry["release_tick"] = tick
                entry["release_time_sec"] = round(tick * DT, 3)
            self.pending[ident] = entry
        self.events.append({"onset_tick": tick, "onset_time_sec": round(tick * DT, 3),
                            "phase": new_phase, "cohort": list(cohort.values())})


def _state(sim, stable_ids):
    rows = []
    for v in sim.vehicles:
        key = id(v)
        if key not in stable_ids or stable_ids[key][0] is not v:
            stable_ids["__count"] = stable_ids.get("__count", 0) + 1
            stable_ids[key] = (v, f"v{stable_ids['__count']:06d}")
        rows.append({"id": stable_ids[key][1], "arm": v.arm, "lane": v.lane,
            "occupied": v.occupied, "speed": v.speed,
            "road_state": sim._road_state(v), "position": v.position,
            "approach_end": v.approach_end, "length": v.length})
    return rows


def audit(seed=1, horizon=300, flow=2400.0, paired_demand=True,
          controller="fixed", actions=(2, 0)):
    if controller not in ("fixed", "train"):
        raise ValueError("controller must be fixed or train")
    temp = tempfile.TemporaryDirectory(prefix="lvl_discharge_") if paired_demand else None
    try:
        demand_path = None
        demand_hash = None
        if temp:
            demand_path = Path(temp.name) / "demand.csv"
            demand_hash = generate_demand(demand_path, seed, horizon, flow)
        sim = LeVanLuongSimulator(seed=seed, flow_per_approach=flow,
            demand_file=demand_path, train_mode=(controller == "train"))
        tracker = DischargeTracker()
        committed = set()
        red_violations = []
        phase_before = sim.phase
        action_index = 0
        stable_ids = {}
        for tick in range(1, horizon + 1):
            if controller == "train" and sim.dqn_waiting_action:
                sim.set_action(actions[action_index % len(actions)])
                action_index += 1
            before = _state(sim, stable_ids)
            was_green = {x["id"] for x in before
                         if x["road_state"][0].endswith("_in") and
                         green_for(x["arm"], phase_before) and
                         x["position"] > x["approach_end"] - x["length"] / 2 - STOP_MARGIN}
            sim.step()
            after = _state(sim, stable_ids)
            tracker.observe(tick, phase_before, sim.phase, before, after)
            live = {x["id"] for x in after}
            committed.intersection_update(live)
            committed.update(was_green)
            for x in after:
                road = x["road_state"][0]
                if road.endswith("_in") and not green_for(x["arm"], sim.phase):
                    front_crossed = x["position"] + x["length"] / 2 + STOP_MARGIN > x["approach_end"] + 1e-8
                    if front_crossed and x["id"] not in committed:
                        red_violations.append({"tick": tick, "vehicle_id": x["id"], "arm": x["arm"]})
            phase_before = sim.phase
        events = tracker.events
        all_entries = [e for event in events for e in event["cohort"]]
        sublane_times = defaultdict(list)
        for event in events:
            for v in event["cohort"]:
                # Each occupied lane gets a release timestamp; the reported headways
                # are measured independently for each approach sublane.
                for lane in range(v["lane"], v["lane"] + v["occupied"]):
                    if v["release_time_sec"] is not None:
                        sublane_times[f'{v["arm"]}:{lane}'].append(v["release_time_sec"])
        headways = {}
        for key, times in sublane_times.items():
            times.sort()
            headways[key] = [round(b-a, 3) for a, b in zip(times, times[1:])]
        summary = {"seed": seed, "horizon_ticks": horizon, "dt_sec": DT,
            "flow_per_approach": flow, "paired_demand": paired_demand,
            "demand_sha256": demand_hash, "controller": controller,
            "actions": list(actions) if controller == "train" else None,
            "red_green_events": len(events), "events": events,
            "cohort_vehicles": len(all_entries),
            "accelerated_same_tick": sum(v["release_tick"] == event["onset_tick"]
                for event in events for v in event["cohort"]),
            "accelerated_within_0_2s": sum(v["release_tick"] is not None and
                v["release_tick"] - event["onset_tick"] <= 1
                for event in events for v in event["cohort"]),
            "accelerated_within_1s": sum(v["release_tick"] is not None and
                v["release_tick"] - event["onset_tick"] <= 5
                for event in events for v in event["cohort"]),
            "unreleased_at_horizon": sum(v["release_tick"] is None for v in all_entries),
            "sublane_release_headways_sec": headways,
            "red_stop_line_violations": len(red_violations),
            "red_stop_line_violation_events": red_violations,
            "created": sim.created, "passed": sim.passed, "active_at_end": len(sim.vehicles)}
        return summary
    finally:
        if temp:
            temp.cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--flow", type=float, default=2400)
    p.add_argument("--unpaired-demand", action="store_true")
    p.add_argument("--controller", choices=("fixed", "train"), default="fixed")
    p.add_argument("--actions", default="2,0", help="Train actions at successive decision ticks; 0=LVL, 2=HDT")
    p.add_argument("--output", type=Path)
    p.add_argument("--enforce", action="store_true")
    p.add_argument("--max-same-tick-fraction", type=float, default=1.0)
    p.add_argument("--max-red-violations", type=int, default=0)
    args = p.parse_args()
    if args.horizon < 0:
        p.error("horizon must be nonnegative")
    try:
        actions = tuple(int(v.strip()) for v in args.actions.split(",") if v.strip())
    except ValueError:
        p.error("--actions must be a comma-separated list of 0 and 2")
    if not actions or any(a not in (0, 2) for a in actions):
        p.error("--actions must contain only 0 and 2")
    row = audit(args.seed, args.horizon, args.flow, not args.unpaired_demand,
                args.controller, actions)
    source = ROOT / "python-simulation" / "simulation" / "python_simulation" / "python_sim.py"
    report = {"python_sim_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": [row]}
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if args.enforce:
        fraction = row["accelerated_same_tick"] / max(1, row["cohort_vehicles"])
        if fraction > args.max_same_tick_fraction or row["red_stop_line_violations"] > args.max_red_violations:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
