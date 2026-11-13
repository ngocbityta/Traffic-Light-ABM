#!/usr/bin/env python3
"""Audit safe-mode vehicle footprints on Le Van Luong routes.

This is an observer: it does not alter simulation state. Sweeps are sampled at
5 cm maximum route spacing and inflate rectangles by half that spacing; this
reduces but does not mathematically eliminate errors from sharp polyline bends.
"""
import argparse
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "python-simulation" / "simulation" / "python_simulation"
sys.path.insert(0, str(REPO))
PYTHON_SIM_ROOT = REPO / "python-simulation" / "simulation"
sys.path.insert(0, str(PYTHON_SIM_ROOT))
from python_simulation.python_sim import LeVanLuongSimulator
from python_simulation.vehicle_geometry import Road, SUBLANE_WIDTH, footprint, polygons_overlap


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Route:
    """Concatenated route arc length; preserves each component road's lane width."""
    def __init__(self, roads, arm, turn):
        self.parts = []
        self.offsets = []
        connector = next(r for r in roads.values() if r["kind"] == "connector" and r["arm"] == arm and r["turn"] == turn)
        ids = (f"{arm}_in", connector["id"])
        ids += (f"{connector['to_arm']}_out",)
        offset = 0.0
        for rid in ids:
            spec = roads[rid]
            road = Road(spec["coordinates"], spec["num_lanes"])
            self.parts.append((rid, road, offset))
            self.offsets.append(offset)
            offset += road.length
        self.length = offset

    def pose(self, progress, shift):
        s = float(progress)
        idx = len(self.parts) - 1
        for i, (_, road, off) in enumerate(self.parts):
            if s <= off + road.length:
                idx = i
                break
        rid, road, off = self.parts[idx]
        # Road.pose uses the same lateral world offset as simulator state.
        return rid, road.pose(s - off, shift)

    def part(self, road_id):
        return next((road, off) for rid, road, off in self.parts if rid == road_id)

    def pose_on_road(self, road_id, road_progress, shift):
        """Pose in a specified road frame, including rebased seam states."""
        road, _ = self.part(road_id)
        return road.pose(road_progress, shift)

    def leg(self, progress):
        return self.pose(progress, 0)[0]


def build_roads(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for f in data["features"]:
        p = f["properties"]
        result[p["id"]] = {**p, "coordinates": f["geometry"]["coordinates"]}
    return result


def vehicle_state(v, routes):
    route = routes[(v.arm, v.to_arm, v.maneuver)]
    # Simulator state and observer.html both store this offset in metres.
    shift = float(getattr(v, "current_lateral_shift", 0.0))
    rid, route_pose = route.pose(v.position, shift)
    sim_road = str(getattr(v, "road", rid))
    try:
        if not hasattr(route, "part"):
            raise StopIteration
        _, road_offset = route.part(sim_road)
        pose = route.pose_on_road(sim_road, v.position - road_offset, shift)
        rid = sim_road
    except StopIteration:
        pose = route_pose
    kind = "car" if v.occupied == 2 else "motorbike"
    return {"id": str(v.demand_row_id if v.demand_row_id >= 0 else id(v)),
            "road": rid, "sim_road": sim_road,
            "arm": v.arm, "to_arm": v.to_arm, "maneuver": v.maneuver,
            "kind": kind, "position": float(v.position), "shift": shift,
            "speed": float(getattr(v, "speed", 0.0)), "lane": int(v.lane), "pose": pose,
            "route": route, "vehicle": v}


def _bbox(poly):
    return (min(x for x, _ in poly), min(y for _, y in poly),
            max(x for x, _ in poly), max(y for _, y in poly))


def _boxes_overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _boxes_overlap_clearance(a, b, clearance):
    return (a[0] <= b[2] + 2 * clearance and b[0] <= a[2] + 2 * clearance and
            a[1] <= b[3] + 2 * clearance and b[1] <= a[3] + 2 * clearance)


def _seam_jump(tick, vehicle_id, route, old_road_id, new_road_id,
               old_lane, new_lane, old_shift, new_shift, seam_progress=None):
    """Measure world-center discontinuity across one actual route seam."""
    old_road = next(road for rid, road, _ in route.parts if rid == old_road_id)
    new_road = next(road for rid, road, _ in route.parts if rid == new_road_id)
    before = old_road.pose(old_road.length, old_shift)
    origin = new_road.pose(0.0, new_road.num_lanes * SUBLANE_WIDTH - SUBLANE_WIDTH / 2.0)
    if seam_progress is None:
        dx, dy = before.x - origin.x, before.y - origin.y
        seam_progress = dx * origin.tx + dy * origin.ty
    after = new_road.pose(seam_progress, new_shift)
    jump = math.hypot(after.x - before.x, after.y - before.y)
    return {"tick": tick, "vehicle_id": vehicle_id,
            "route_id": f"{route.parts[0][0]}->{route.parts[1][0]}->{route.parts[2][0]}",
            "from_road": old_road_id, "to_road": new_road_id,
            "seam": f"{old_road_id}->{new_road_id}",
            "old_lane": old_lane, "new_lane": new_lane,
            "old_lateral_shift_m": old_shift, "new_lateral_shift_m": new_shift,
            "old_center": {"x": before.x, "y": before.y},
            "new_center": {"x": after.x, "y": after.y},
            "center_jump_m": jump}


def _audit_id(v, spawn_ids):
    if v.demand_row_id >= 0:
        return f"demand-{v.demand_row_id}"
    identity = id(v)
    if identity not in spawn_ids:
        spawn_ids[identity] = f"spawn-{len(spawn_ids) + 1}"
    return spawn_ids[identity]


def audit(seed=1, cycles=100, mode="safe", clearance=0.0, roads_path=None, demand_file=None):
    roads_path = Path(roads_path or ROOT / "includes" / "roads.geojson")
    sim = LeVanLuongSimulator(seed=seed, mode=mode, roads_path=roads_path, demand_file=demand_file)
    roads = build_roads(roads_path)
    route_cache = {}
    routes = {}
    static_events, swept_events, seam_jumps = [], [], []
    previous = {}
    spawn_ids = {}
    for _ in range(cycles):
        before = {}
        for v in sim.vehicles:
            key = (v.arm, v.to_arm, v.maneuver)
            if key not in route_cache:
                route_cache[key] = Route(roads, v.arm, v.maneuver)
            before[_audit_id(v, spawn_ids)] = vehicle_state(v, route_cache)
        sim.step()
        states = []
        for v in sim.vehicles:
            key = (v.arm, v.to_arm, v.maneuver)
            if key not in route_cache:
                route_cache[key] = Route(roads, v.arm, v.maneuver)
            state = vehicle_state(v, route_cache)
            state["id"] = _audit_id(v, spawn_ids)
            states.append(state)
        for state in states:
            old = before.get(state["id"])
            if old and old["sim_road"] != state["sim_road"]:
                route = state["route"]
                route_roads = [item[0] for item in route.parts]
                if (old["sim_road"] in route_roads and state["sim_road"] in route_roads and
                        abs(route_roads.index(old["sim_road"]) - route_roads.index(state["sim_road"])) == 1):
                    seam_jumps.append(_seam_jump(
                        sim.cycle, state["id"], route, old["sim_road"], state["sim_road"],
                        old["lane"], state["lane"],
                        old["shift"], state["vehicle"].seam_shift,
                        state["vehicle"].seam_progress))
        polygons = [footprint(s["pose"], s["kind"]) for s in states]
        boxes = [_bbox(poly) for poly in polygons]
        # Sweep-and-prune broad phase on the x axis avoids quadratic SAT work
        # for the full traffic field; y-bounds discard the remaining far pairs.
        order = sorted(range(len(states)), key=lambda i: boxes[i][0])
        candidates = []
        for oi, i in enumerate(order):
            for j in order[oi + 1:]:
                if boxes[j][0] > boxes[i][2] + 2 * clearance:
                    break
                if _boxes_overlap_clearance(boxes[i], boxes[j], clearance):
                    candidates.append((i, j))
        for i, j in candidates:
            a, b = states[i], states[j]
            pair = tuple(sorted((a["id"], b["id"])))
            if polygons_overlap(polygons[i], polygons[j], clearance):
                static_events.append(_event(sim.cycle, a, b, "static"))
        # Also consider pairs whose current boxes do not touch: their broad
        # phase is the swept route AABB, tested below against previous states.
        curr = {s["id"]: s for s in states}
        # Swept broad phase: each moving center's endpoint envelope expanded
        # by the vehicle circumradius, then sweep-and-prune on x.
        moving = []
        for s in states:
            p = previous.get(s["id"])
            if not p:
                continue
            radius = 2.25 if s["kind"] == "car" else 0.95
            pa = p["pose"]
            moving.append((s, p, (min(pa.x, s["pose"].x)-radius-clearance,
                                  min(pa.y, s["pose"].y)-radius-clearance,
                                  max(pa.x, s["pose"].x)+radius+clearance,
                                  max(pa.y, s["pose"].y)+radius+clearance)))
        moving.sort(key=lambda item: item[2][0])
        for i, (a, pa, ba) in enumerate(moving):
            for b, pb, bb in moving[i+1:]:
                if bb[0] > ba[2]:
                    break
                if not _boxes_overlap(ba, bb):
                    continue
                if _sampled_sweep(pa, a, pb, b, clearance):
                    swept_events.append(_event(sim.cycle, a, b, "swept_sampled"))
        previous = curr
    first = (static_events + swept_events)
    first.sort(key=lambda e: (e["tick"], e["type"], e["vehicle_a"], e["vehicle_b"]))
    largest_seam = max(seam_jumps, key=lambda x: x["center_jump_m"], default=None)
    def event_groups(items):
        groups = {"road_road": 0, "connector_connector": 0, "road_connector": 0}
        for event in items:
            a = event["road_a"].split("_")
            b = event["road_b"].split("_")
            ac = len(a) > 1 and len(a[1]) == 2
            bc = len(b) > 1 and len(b[1]) == 2
            key = "connector_connector" if ac and bc else (
                "road_connector" if ac or bc else "road_road")
            groups[key] += 1
        return groups
    seam_summary = {}
    for event in seam_jumps:
        bucket = seam_summary.setdefault(event["seam"], {"count": 0, "max_center_jump_m": 0.0})
        bucket["count"] += 1
        bucket["max_center_jump_m"] = max(bucket["max_center_jump_m"], event["center_jump_m"])
    final_road_counts = {}
    for vehicle in sim.vehicles:
        road_id = str(getattr(vehicle, "road", ""))
        final_road_counts[road_id] = final_road_counts.get(road_id, 0) + 1
    return {"schema_version": 1, "seed": seed, "horizon_ticks": cycles, "mode": mode,
            "simulation_coverage": {"created": sim.created, "passed": sim.passed,
                                    "active": len(sim.vehicles),
                                    "active_vehicles_by_road": final_road_counts,
                                    "observed_seam_transitions": len(seam_jumps)},
            "scope": {"vehicles": "safe-mode simulation vehicles only", "roads": "approach, connector, exit, including seams via concatenated route arc length",
                      "geometry": "oriented car 4.5x1.7m / motorbike 1.9x0.7m rectangles", "clearance_m": clearance,
                      "sweep": "sampled at <=0.05m route distance with half-step rectangle inflation; not a continuous proof",
                      "baseline_expectation": "Existing safe mode has not been validated for connector footprint collisions; any event is reported as a baseline violation, with no simulator correction applied",
                      "limitations": "road polylines approximate geometry; sampled sweep may miss behavior between samples or at heading discontinuities"},
            "source_hashes": {"python_sim.py": sha256(inspect.getsourcefile(LeVanLuongSimulator)), "vehicle_geometry.py": sha256(inspect.getsourcefile(footprint)), "roads.geojson": sha256(roads_path)},
            "demand_file_sha256": sha256(demand_file) if demand_file else None,
            "static_event_count": len(static_events), "swept_event_count": len(swept_events),
            "static_event_groups": event_groups(static_events),
            "swept_event_groups": event_groups(swept_events),
            "seam_transition_count": len(seam_jumps),
            "max_seam_center_jump_m": max((x["center_jump_m"] for x in seam_jumps), default=0.0),
            "first_seam_jump": seam_jumps[0] if seam_jumps else None,
            "largest_seam_jump": largest_seam,
            "seam_summary": seam_summary,
            "seam_jumps": seam_jumps,
            "first_event": first[0] if first else None,
            "events": {"static": static_events, "swept_sampled": swept_events}}


def _sampled_sweep(pa, a, pb, b, clearance):
    # Same route and stable route coordinates are essential across road seams.
    if pa["arm"] != a["arm"] or pa["maneuver"] != a["maneuver"] or pb["arm"] != b["arm"] or pb["maneuver"] != b["maneuver"]:
        return False
    path_a = _sweep_path(pa, a)
    path_b = _sweep_path(pb, b)
    # Rebasing lateral shift at a seam changes coordinates, not the vehicle's
    # physical lateral position. Do not count that coordinate jump as travel.
    travel = path_a["travel"] + path_b["travel"]
    n = max(1, math.ceil(travel / 0.05))
    pad = travel / (2 * n)
    for i in range(n + 1):
        t = i / n
        xa = path_a["pose_at"](t)
        xb = path_b["pose_at"](t)
        la, wa = (4.5, 1.7) if a["kind"] == "car" else (1.9, .7)
        lb, wb = (4.5, 1.7) if b["kind"] == "car" else (1.9, .7)
        if polygons_overlap(footprint(xa, length=la + 2*pad, width=wa + 2*pad),
                            footprint(xb, length=lb + 2*pad, width=wb + 2*pad), clearance):
            return True
    return False


def _sweep_path(before, after):
    """Return a tick trajectory with an explicit piecewise road seam.

    `position` is route progress before a crossing, but the simulator rebases it
    by `seam_progress` at the crossing. `current_lateral_shift` is likewise
    rebased into the new road frame. Interpolating either coordinate across the
    whole tick invents a path through space. The vehicle's seam metadata records
    the new-road local progress and shift, which lets the observer reconstruct
    the two physical legs while using the same Road.pose mapping as the model.
    """
    route = after["route"]
    before_pose = before.get("pose")
    if before_pose is None:
        before_pose = route.pose(before["position"], before["shift"])[1]
    after_pose = after.get("pose")
    if after_pose is None:
        after_pose = route.pose(after["position"], after["shift"])[1]
    vehicle = after.get("vehicle")
    old_road = before.get("sim_road", before.get("road"))
    new_road = after.get("sim_road", after.get("road"))
    crossed = (old_road != new_road and
               vehicle is not None and
               getattr(vehicle, "seam_from_road", "") == old_road and
               getattr(vehicle, "seam_to_road", "") == new_road)
    if crossed:
        try:
            old_geom, old_offset = route.part(old_road)
            _, new_offset = route.part(new_road)
        except StopIteration:
            crossed = False
    if crossed:
        old_boundary = old_offset + old_geom.length
        seam_progress = float(vehicle.seam_progress)
        seam_shift = float(vehicle.seam_shift)
        seam_position = new_offset + seam_progress
        before_distance = max(0.0, old_boundary - before["position"])
        after_distance = max(0.0, after["position"] - seam_position)
        total_distance = before_distance + after_distance
        seam_t = (before_distance / total_distance if total_distance > 1e-12
                  else 0.5)
        seam_t = min(1.0, max(0.0, seam_t))
        # Physical travel excludes the discontinuous coordinate rebase. Any
        # shift change after the seam is a real within-tick lane movement.
        travel = total_distance + abs(after["shift"] - seam_shift)

        def pose_at(t):
            if t <= 0.0:
                return before_pose
            if t >= 1.0:
                return after_pose
            if t <= seam_t:
                u = t / seam_t if seam_t > 0 else 1.0
                progress = before["position"] + before_distance * u
                return route.pose_on_road(old_road, progress - old_offset,
                                          before["shift"])
            u = ((t - seam_t) / (1.0 - seam_t)
                 if seam_t < 1.0 else 1.0)
            local_progress = seam_progress + after_distance * u
            shift = seam_shift + (after["shift"] - seam_shift) * u
            return route.pose_on_road(new_road, local_progress, shift)

        return {"travel": travel, "pose_at": pose_at}

    travel = (abs(after["position"] - before["position"]) +
              abs(after["shift"] - before["shift"]))

    def pose_at(t):
        if t <= 0.0:
            return before_pose
        if t >= 1.0:
            return after_pose
        progress = before["position"] + (after["position"] - before["position"]) * t
        shift = before["shift"] + (after["shift"] - before["shift"]) * t
        return route.pose(progress, shift)[1]

    return {"travel": travel, "pose_at": pose_at}


def _event(tick, a, b, kind):
    return {"tick": tick, "type": kind, "vehicle_a": a["id"], "vehicle_b": b["id"],
            "road_a": a["road"], "road_b": b["road"], "pose_a": _pose_json(a["pose"]), "pose_b": _pose_json(b["pose"])}


def _pose_json(p):
    return {"x": p.x, "y": p.y, "heading_rad": p.heading, "progress_m": p.progress}


def enforce_failed(report, min_passed=1):
    """Fail an enforced geometry gate on events or insufficient traffic coverage."""
    coverage = report.get("simulation_coverage", {})
    return (bool(report["static_event_count"] or report["swept_event_count"]) or
            coverage.get("passed", 0) < min_passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--mode", default="safe", choices=("safe", "gama_2025_6_4"))
    parser.add_argument("--clearance", type=float, default=0.0)
    parser.add_argument("--roads", type=Path, default=ROOT / "includes" / "roads.geojson")
    parser.add_argument("--demand", type=Path, help="optional six-column demand replay CSV")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce", action="store_true", help="exit 1 if any event is found")
    parser.add_argument("--min-passed", type=int, default=1,
                        help="with --enforce, require at least this many vehicles to pass (default: 1)")
    args = parser.parse_args()
    if args.min_passed < 0:
        parser.error("--min-passed must be non-negative")
    report = audit(args.seed, args.cycles, args.mode, args.clearance, args.roads, args.demand)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return int(args.enforce and enforce_failed(report, args.min_passed))


if __name__ == "__main__":
    raise SystemExit(main())
