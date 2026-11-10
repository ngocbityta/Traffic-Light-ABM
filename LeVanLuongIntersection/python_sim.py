#!/usr/bin/env python3
"""GeoJSON-backed Python simulator for the Le Van Luong GAML layout.

Road lengths, sublane counts, turns, signal cycle and arrival probabilities
come directly from the current GeoJSON and LeVanLuong.gaml.  Vehicle motion
uses ported IDM and MOBIL rules; paired benchmarks still quantify remaining
individual-vehicle differences.
"""

import argparse
import bisect
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARMS = ("NE", "SE", "SW", "NW")
DT = 0.2


def road_lengths(path=ROOT / "includes" / "roads.geojson"):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    roads = {}
    for feature in source["features"]:
        props = feature["properties"]
        points = feature["geometry"]["coordinates"]
        length = sum(math.dist(a[:2], b[:2]) for a, b in zip(points, points[1:]))
        roads[props["id"]] = {**props, "length": length}
    return roads


@dataclass(slots=True)
class Vehicle:
    arm: str
    to_arm: str
    maneuver: str
    lane: int
    occupied: int
    length: float
    desired: float
    accel: float
    decel: float
    gap: float
    headway: float
    approach_end: float
    connector_end: float
    route_end: float
    demand_row_id: int = -1
    position: float = 0.0
    speed: float = 0.0
    time_since_lane_change: float = 0.0
    leading_distance: float = math.inf
    leading_speed: float = 0.0
    road: str = ""
    current_lateral_shift: float = 0.0
    start_lateral_shift: float = 0.0
    target_lateral_shift: float = 0.0
    lane_change_progress: float = 1.0
    lateral_duration: float = 0.8

    def bezier_control_points(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Return (P1, P2, P3, P4) cubic Bezier control points per Naseri et al. (2017) Section 3.1."""
        speed = max(1.0, self.speed)
        length = speed * max(0.1, self.lateral_duration)
        p1 = (0.0, self.start_lateral_shift)
        p2 = (length / 3.0, self.start_lateral_shift)
        p3 = (2.0 * length / 3.0, self.target_lateral_shift)
        p4 = (length, self.target_lateral_shift)
        return p1, p2, p3, p4

    def bezier_curvature_radius(self) -> float:
        """Minimum curvature radius r = v^2 / a_allow per Naseri et al. (2017) Section 3.1."""
        speed = max(1.0, self.speed)
        a_allow = 2.0  # Comfort acceleration limit (Wu et al. 2009, Naseri et al. 2017)
        return speed * speed / a_allow


class LeVanLuongSimulator:
    def __init__(self, seed=1, flow_per_approach=2400.0,
                 motorbike_ratio=.85, p_left=.2, p_right=.2,
                 green_lvl=50.0, green_hdt=40.0, all_red=3.0,
                 roads_path=ROOT / "includes" / "roads.geojson",
                 demand_file=None, train_mode=False):
        if (flow_per_approach < 0 or not 0 <= motorbike_ratio <= 1 or
                min(p_left, p_right, green_lvl, green_hdt, all_red) < 0 or
                p_left + p_right > 1 or flow_per_approach / 3600 * DT > 1):
            raise ValueError("Invalid flow, vehicle mix, turn split, or signal timing")
        self.rng = random.Random(seed)
        self.roads = road_lengths(roads_path)
        self.flow_per_approach = flow_per_approach
        self.motorbike_ratio = motorbike_ratio
        self.p_left = p_left
        self.p_right = p_right
        self.green_lvl = green_lvl
        self.green_hdt = green_hdt
        self.all_red = all_red
        self.cycle = 0
        self.phase = "LVL"
        self.vehicles = []
        self.created = self.passed = 0
        self.front_row_max = {arm: 0 for arm in ARMS}
        self.queue_time_integral = 0.0
        self.system_time_integral = 0.0
        self.weighted_delay_integral = 0.0
        self.green_ticks = 0
        self.train_mode = bool(train_mode)
        self.dqn_waiting_action = False
        self.dqn_action = -1
        self.next_decision_in_cycles = 50
        self.red_ticks_left = 0
        self.pending_phase = "LVL"
        self._rows = []
        self._cursor = 0
        self._replay = demand_file is not None
        if self._replay:
            with Path(demand_file).open(newline="", encoding="utf-8") as stream:
                for raw in csv.reader(stream):
                    if raw:
                        if len(raw) != 6:
                            raise ValueError(f"Invalid demand row: {raw}")
                        row = (int(raw[0]), int(raw[1]), int(raw[2]),
                               float(raw[3]), int(raw[4]), int(raw[5]))
                        if (not 0 <= row[1] < 4 or
                                row[2] not in (0, 1) or not 0 <= row[4] <= 2):
                            raise ValueError(f"Invalid demand row: {raw}")
                        self._rows.append(row)
            if self._rows != sorted(self._rows, key=lambda row: (row[0], row[1])):
                raise ValueError("Demand rows must be ordered by cycle and arm")

    def _phase(self):
        # GAMA's ``time`` at cycle n is (n - 1) * step inside the reflex.
        t = max(0, self.cycle - 1) * DT
        length = self.green_lvl + self.green_hdt + 2 * self.all_red
        if length <= 0:
            return "none"
        t %= length
        if t < self.green_lvl:
            return "LVL"
        if t < self.green_lvl + self.all_red:
            return "none"
        if t < self.green_lvl + self.all_red + self.green_hdt:
            return "HDT"
        return "none"

    def set_action(self, action):
        if not self.train_mode or not self.dqn_waiting_action or action not in (0, 2):
            raise ValueError("Action requires a Le Van Luong decision tick and phase 0 or 2")
        self.dqn_action = action
        self.dqn_waiting_action = False

    def _advance_signal(self):
        if not self.train_mode:
            self.phase = self._phase()
            if self.phase in ("LVL", "HDT"):
                self.green_ticks += 1
            else:
                self.green_ticks = 0
            return
        if self.dqn_action >= 0:
            wanted = "LVL" if self.dqn_action == 0 else "HDT"
            if wanted != self.phase and self.red_ticks_left == 0:
                self.pending_phase = wanted
                self.red_ticks_left = 15
            self.dqn_action = -1
        if self.red_ticks_left > 0:
            self.phase = "none"
            self.red_ticks_left -= 1
            if self.red_ticks_left == 0:
                self.phase = self.pending_phase
        if self.phase in ("LVL", "HDT"):
            self.green_ticks += 1
        else:
            self.green_ticks = 0
        if self.cycle > 0 and self.cycle % 50 == 0:
            self.dqn_waiting_action = True
        self.next_decision_in_cycles = (1 if self.dqn_waiting_action else
                                        50 - self.cycle % 50)

    def _spawn(self):
        if self._replay:
            while self._cursor < len(self._rows) and self._rows[self._cursor][0] <= self.cycle:
                _, origin, car, speed, turn, lane = self._rows[self._cursor]
                self._create_vehicle(ARMS[origin], ("left", "right", "through")[turn],
                                     bool(car), lane, speed, self._cursor)
                self._cursor += 1
            return
        for arm in ARMS:
            if self.rng.random() >= self.flow_per_approach / 3600 * DT:
                continue
            draw = self.rng.random()
            turn = "left" if draw < self.p_left else (
                "right" if draw < self.p_left + self.p_right else "through")
            incoming = self.roads[f"{arm}_in"]
            is_car = self.rng.random() < 1 - self.motorbike_ratio
            occupied = 2 if is_car else 1
            lane = self.rng.randrange(incoming["num_lanes"] - occupied + 1)
            speed_kmh = max(1.0, self.rng.gauss(50 if is_car else 40, 5))
            self._create_vehicle(arm, turn, is_car, lane, speed_kmh)

    def _create_vehicle(self, arm, turn, is_car, lane, speed_kmh, demand_row_id=-1):
        connector = next(road for road in self.roads.values()
                         if road["kind"] == "connector" and
                         road["arm"] == arm and road["turn"] == turn)
        target = connector["to_arm"]
        incoming = self.roads[f"{arm}_in"]
        outgoing = self.roads[f"{target}_out"]
        occupied = 2 if is_car else 1
        if not 0 <= lane <= incoming["num_lanes"] - occupied or speed_kmh <= 0:
            raise ValueError("Invalid lane or requested speed")
        approach_end = incoming["length"]
        connector_end = approach_end + connector["length"]
        initial_shift = (lane + (occupied - 1) / 2.0) * 0.8
        self.vehicles.append(Vehicle(
            arm=arm, to_arm=target, maneuver=turn, lane=lane, occupied=occupied,
            length=4.5 if is_car else 1.9, desired=speed_kmh / 3.6,
            accel=2.5 if is_car else 2.0, decel=3.0,
            gap=1.0 if is_car else .3, headway=1.2 if is_car else .6,
            approach_end=approach_end,
            connector_end=connector_end,
            route_end=connector_end + outgoing["length"],
            road=f"{arm}_in",
            demand_row_id=demand_row_id,
            current_lateral_shift=initial_shift,
            start_lateral_shift=initial_shift,
            target_lateral_shift=initial_shift,
            lane_change_progress=1.0,
            lateral_duration=0.6 if not is_car else 1.2))
        self.created += 1

    def _move(self):
        # GAMA iterates agents in insertion order. Each agent performs its
        # lane choice and movement before the next agent reads its neighbors.
        lane_maps = {}
        self._free_idm_cache = {}
        self._neighbor_road_state = {}
        for vehicle in self.vehicles:
            road, progress = self._road_state(vehicle)
            self._neighbor_road_state[id(vehicle)] = (road, progress)
            if road not in lane_maps:
                lane_maps[road] = [[] for _ in range(self.roads[road]["num_lanes"])]
            entry = (progress, id(vehicle), vehicle)
            for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                bisect.insort(lane_maps[road][sub], entry)
        survivors = []
        for vehicle in self.vehicles:
            old_road, old_progress = self._neighbor_road_state[id(vehicle)]
            is_in_road = old_road.endswith("_in")
            green = (
                (vehicle.arm in ("NE", "SW") and self.phase == "LVL") or
                (vehicle.arm in ("SE", "NW") and self.phase == "HDT")
            ) if is_in_road else True

            # GAMA DrivingSkill.moveAcrossSegments:
            # If distanceToNextPoint < 0.01 at the start of the step, the agent
            # snaps directly to the node (rem = 0.0). If red (!readyToCross),
            # speed is set to 0.0 and movement terminates without lane change.
            if is_in_road and not green and vehicle.approach_end - vehicle.position < 0.01:
                vehicle.position = vehicle.approach_end
                vehicle.speed = 0.0
                self._free_idm_cache.pop(id(vehicle), None)
                if old_progress != vehicle.approach_end:
                    old_entry = (old_progress, id(vehicle), vehicle)
                    for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                        entries = lane_maps[old_road][sub]
                        index = bisect.bisect_left(entries, old_entry)
                        if index < len(entries) and entries[index] == old_entry:
                            entries.pop(index)
                    new_entry = (vehicle.approach_end, id(vehicle), vehicle)
                    for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                        bisect.insort(lane_maps[old_road][sub], new_entry)
                    self._neighbor_road_state[id(vehicle)] = (old_road, vehicle.approach_end)
                survivors.append(vehicle)
                continue

            self._choose_lane(vehicle, lane_maps)
            move_lane = vehicle.lane
            leader, gap, _, _ = self._neighbors(vehicle, vehicle.lane, lane_maps)
            vehicle.leading_distance = gap
            vehicle.leading_speed = leader.speed if leader else 0.0
            acceleration = self._idm(vehicle, gap, leader.speed if leader else 0.0)
            initial_speed = vehicle.speed
            predicted_speed = initial_speed + acceleration * DT
            if predicted_speed < 0:
                distance = -0.5 * initial_speed * initial_speed / acceleration
                vehicle.speed = 0.0
            else:
                distance = initial_speed * DT + 0.5 * acceleration * DT * DT
                vehicle.speed = predicted_speed
            if math.isfinite(gap):
                distance = min(distance, max(0.0, gap))
            boundary = (vehicle.approach_end if is_in_road else
                        vehicle.connector_end if old_road not in
                        (f"{vehicle.to_arm}_out",) else vehicle.route_end)
            to_boundary = boundary - vehicle.position
            can_cross = green if is_in_road else True
            if (can_cross and boundary < vehicle.route_end and
                    distance >= to_boundary and predicted_speed > 0):
                # GAMA reaches the node using the predicted old-road speed,
                # then enters the new road, evaluates MOBIL lane choice, and
                # computes IDM with the new road's parameters.
                remaining = max(0.0, DT - to_boundary / predicted_speed)
                vehicle.position = boundary
                old_entry = (old_progress, id(vehicle), vehicle)
                for sub in range(move_lane, move_lane + vehicle.occupied):
                    entries = lane_maps[old_road][sub]
                    index = bisect.bisect_left(entries, old_entry)
                    if index < len(entries) and entries[index] == old_entry:
                        entries.pop(index)
                if is_in_road:
                    vehicle.road = f"{vehicle.arm}_{vehicle.to_arm}"
                else:
                    vehicle.road = f"{vehicle.to_arm}_out"
                next_road = vehicle.road
                vehicle.lane = min(vehicle.lane,
                                   self.roads[next_road]["num_lanes"] - vehicle.occupied)
                new_shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * 0.8
                vehicle.start_lateral_shift = vehicle.target_lateral_shift = new_shift
                vehicle.current_lateral_shift = new_shift
                vehicle.lane_change_progress = 1.0
                if next_road not in lane_maps:
                    lane_maps[next_road] = [[] for _ in range(self.roads[next_road]["num_lanes"])]
                self._neighbor_road_state[id(vehicle)] = (next_road, 0.0)
                self._free_idm_cache.pop(id(vehicle), None)
                new_entry = (0.0, id(vehicle), vehicle)
                for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                    bisect.insort(lane_maps[next_road][sub], new_entry)
                self._choose_lane(vehicle, lane_maps)
                new_leader, new_gap, _, _ = self._neighbors(vehicle, vehicle.lane, lane_maps)
                vehicle.leading_distance = new_gap
                vehicle.leading_speed = new_leader.speed if new_leader else 0.0
                crossing_gap = new_gap if new_leader else math.inf
                new_acceleration = self._idm(
                    vehicle, crossing_gap, new_leader.speed if new_leader else 0.0)
                if initial_speed + new_acceleration * remaining < 0:
                    new_distance = -0.5 * initial_speed * initial_speed / new_acceleration
                    vehicle.speed = 0.0
                else:
                    new_distance = (initial_speed * remaining +
                                    0.5 * new_acceleration * remaining * remaining)
                    vehicle.speed = max(0.0, initial_speed + new_acceleration * DT)
                if math.isfinite(new_gap):
                    new_distance = min(new_distance, max(0.0, new_gap))
                vehicle.position += max(0.0, new_distance)
                curr_entry = (0.0, id(vehicle), vehicle)
                for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                    entries = lane_maps[next_road][sub]
                    index = bisect.bisect_left(entries, curr_entry)
                    if index < len(entries) and entries[index] == curr_entry:
                        entries.pop(index)
                final_entry = (new_distance, id(vehicle), vehicle)
                for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                    bisect.insort(lane_maps[next_road][sub], final_entry)
                self._neighbor_road_state[id(vehicle)] = (next_road, new_distance)
                new_road, new_entry = next_road, final_entry
            else:
                vehicle.position += max(0.0, distance)
                if is_in_road and not green and vehicle.position > vehicle.approach_end:
                    vehicle.position = vehicle.approach_end
                    vehicle.speed = 0.0
                self._free_idm_cache.pop(id(vehicle), None)
                old_entry = (old_progress, id(vehicle), vehicle)
                for sub in range(move_lane, move_lane + vehicle.occupied):
                    entries = lane_maps[old_road][sub]
                    index = bisect.bisect_left(entries, old_entry)
                    if index < len(entries) and entries[index] == old_entry:
                        entries.pop(index)
                new_road, new_progress = self._road_state(vehicle)
                vehicle.road = new_road
                self._neighbor_road_state[id(vehicle)] = (new_road, new_progress)
                if new_road not in lane_maps:
                    lane_maps[new_road] = [[] for _ in range(self.roads[new_road]["num_lanes"])]
                new_entry = (new_progress, id(vehicle), vehicle)
                for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                    bisect.insort(lane_maps[new_road][sub], new_entry)
            if vehicle.position >= vehicle.route_end - 0.01:
                self.passed += 1
                for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                    entries = lane_maps[new_road][sub]
                    index = bisect.bisect_left(entries, new_entry)
                    if index < len(entries) and entries[index] == new_entry:
                        entries.pop(index)
            else:
                if vehicle.lane_change_progress < 1.0:
                    vehicle.lane_change_progress = min(
                        1.0, vehicle.lane_change_progress + DT / max(0.001, vehicle.lateral_duration))
                    u = vehicle.lane_change_progress
                    # Cubic Bezier curve per Naseri et al. (2017) Section 3.1:
                    # y(u) = start + (target - start) * (3u^2 - 2u^3)
                    bezier_u = 3.0 * u * u - 2.0 * u * u * u
                    vehicle.current_lateral_shift = (vehicle.start_lateral_shift +
                        (vehicle.target_lateral_shift - vehicle.start_lateral_shift) * bezier_u)
                else:
                    vehicle.current_lateral_shift = vehicle.target_lateral_shift
                survivors.append(vehicle)
        self.vehicles = survivors

    def _choose_lane(self, vehicle, lane_maps):
        road, progress = self._neighbor_road_state[id(vehicle)]
        old_leader, old_gap, old_follower, _ = self._neighbors(
            vehicle, vehicle.lane, lane_maps)
        overlap = old_gap < 0
        if not overlap and vehicle.time_since_lane_change < 4.0:
            vehicle.time_since_lane_change += DT
            vehicle.leading_distance = old_gap
            vehicle.leading_speed = old_leader.speed if old_leader else 0.0
            return
        old_leader_speed = old_leader.speed if old_leader else 0.0
        old_acc = (-sys.float_info.max if overlap else
                   self._idm(vehicle, old_gap, old_leader_speed))
        vehicle.leading_distance = old_gap
        vehicle.leading_speed = old_leader_speed
        best_lane, best_gain = vehicle.lane, 0.0
        best_leader, best_front_gap = old_leader, old_gap
        for lane in range(self.roads[road]["num_lanes"] - vehicle.occupied + 1):
            if lane == vehicle.lane:
                continue
            leader, front_gap, follower, back_gap = self._neighbors(vehicle, lane, lane_maps)
            if front_gap < 0 or back_gap < 0:
                continue
            leader_speed = leader.speed if leader else 0.0
            target_acc = self._idm(vehicle, front_gap, leader_speed)
            follower_delta = 0.0
            if follower is not None and follower is not old_follower:
                # GAMA MOBIL: candidate follower delta is evaluated with
                # priorGap = backGap + vehicleLength + frontGap.
                # If candidate follower already has a closer leader on its current
                # lane (leading_distance < back_gap), follower_delta is 0.
                if getattr(follower, "leading_distance", math.inf) >= back_gap:
                    prior_gap = back_gap + vehicle.length + front_gap
                    after = self._idm(follower, back_gap, vehicle.speed)
                    if after <= -4.0:
                        continue
                    before = self._idm(follower, prior_gap, leader_speed)
                    follower_delta = after - before
            gain = target_acc - old_acc + 0.1 * follower_delta + (0.25 if lane < vehicle.lane else -0.25)
            if gain > max(0.05, best_gain):
                best_lane, best_gain = lane, gain
                best_leader, best_front_gap = leader, front_gap
        if best_lane != vehicle.lane:
            entry = (progress, id(vehicle), vehicle)
            for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                entries = lane_maps[road][sub]
                index = bisect.bisect_left(entries, entry)
                if index < len(entries) and entries[index] == entry:
                    entries.pop(index)
            for sub in range(best_lane, best_lane + vehicle.occupied):
                bisect.insort(lane_maps[road][sub], entry)
            vehicle.lane = best_lane
            vehicle.time_since_lane_change = 0.0
            vehicle.leading_distance = best_front_gap
            vehicle.leading_speed = best_leader.speed if best_leader else 0.0
            # Smooth Bézier lateral transition per Naseri et al. (2017) Section 3.1
            vehicle.start_lateral_shift = vehicle.current_lateral_shift
            vehicle.target_lateral_shift = (best_lane + (vehicle.occupied - 1) / 2.0) * 0.8
            vehicle.lane_change_progress = 0.0
            lateral_dist = abs(vehicle.target_lateral_shift - vehicle.start_lateral_shift)
            # Paper Eq: a_allow = 2.0 m/s^2, r = v^2 / a_allow, T_lc = sqrt(6 * W / a_allow)
            a_allow = 2.0
            base_duration = math.sqrt(6.0 * max(0.4, lateral_dist) / a_allow)
            multiplier = 0.6 if vehicle.occupied == 1 else 1.0
            vehicle.lateral_duration = max(0.4, base_duration * multiplier)
        else:
            vehicle.time_since_lane_change += DT

    def _road_state(self, vehicle):
        road = getattr(vehicle, "road", "")
        if road:
            if road == f"{vehicle.arm}_in" and vehicle.position > vehicle.approach_end:
                road = vehicle.road = f"{vehicle.arm}_{vehicle.to_arm}"
            elif road == f"{vehicle.arm}_{vehicle.to_arm}" and vehicle.position > vehicle.connector_end:
                road = vehicle.road = f"{vehicle.to_arm}_out"
        if not road:
            if vehicle.position <= vehicle.approach_end:
                road = vehicle.road = f"{vehicle.arm}_in"
            elif vehicle.position <= vehicle.connector_end:
                road = vehicle.road = f"{vehicle.arm}_{vehicle.to_arm}"
            else:
                road = vehicle.road = f"{vehicle.to_arm}_out"
        if road == f"{vehicle.arm}_in":
            return road, vehicle.position
        if road == f"{vehicle.arm}_{vehicle.to_arm}":
            return road, vehicle.position - vehicle.approach_end
        return road, vehicle.position - vehicle.connector_end

    @staticmethod
    def _overlap(a, a_lane, b):
        return a_lane < b.lane + b.occupied and b.lane < a_lane + a.occupied

    def _neighbors(self, vehicle, lane, lane_maps):
        road, position = self._neighbor_road_state[id(vehicle)]
        leader = follower = None
        front = back = math.inf
        front_distance = back_distance = math.inf
        for sublane in range(lane, lane + vehicle.occupied):
            entries = lane_maps[road][sublane]
            index = bisect.bisect_left(entries, (position, -1))
            ahead = index
            while ahead < len(entries) and entries[ahead][2] is vehicle:
                ahead += 1
            if ahead < len(entries):
                other_position, _, other = entries[ahead]
                distance = other_position - position
                if distance < front_distance:
                    front_distance = distance
                    leader = other
                    front = distance - (vehicle.length + other.length) / 2
            behind = index - 1
            while behind >= 0 and entries[behind][2] is vehicle:
                behind -= 1
            if behind >= 0:
                other_position, _, other = entries[behind]
                distance = position - other_position
                if distance < back_distance:
                    back_distance = distance
                    follower = other
                    back = distance - (vehicle.length + other.length) / 2
        if follower is None and road == f"{vehicle.arm}_{vehicle.to_arm}":
            # On a connector, GAMA's follower search crosses its source node
            # and checks vehicles approaching on the incoming road.
            source_road = f"{vehicle.arm}_in"
            source = self.roads[source_road]
            for sublane in range(lane, lane + vehicle.occupied):
                if sublane >= source["num_lanes"]:
                    continue
                entries = lane_maps.get(source_road, [])[sublane:sublane + 1]
                if not entries or not entries[0]:
                    continue
                ordered = entries[0]
                index = bisect.bisect_left(ordered, (source["length"], math.inf))
                if not index:
                    continue
                other_position, _, other = ordered[index - 1]
                distance = source["length"] - other_position + position
                if distance < back_distance:
                    back_distance = distance
                    follower = other
                    back = distance - (vehicle.length + other.length) / 2
        if leader is None:
            next_road = None
            if road == f"{vehicle.arm}_in":
                green = (vehicle.arm in ("NE", "SW") and self.phase == "LVL") or (
                    vehicle.arm in ("SE", "NW") and self.phase == "HDT")
                if green:
                    next_road = f"{vehicle.arm}_{vehicle.to_arm}"
                else:
                    front = vehicle.approach_end - vehicle.position + vehicle.gap
            elif road == f"{vehicle.arm}_{vehicle.to_arm}":
                next_road = f"{vehicle.to_arm}_out"
            elif position > 0:
                front = vehicle.route_end - vehicle.position + vehicle.gap
            if next_road in lane_maps:
                next_lanes = self.roads[next_road]["num_lanes"]
                next_lane = min(lane, next_lanes - vehicle.occupied)
                for sublane in range(next_lane, next_lane + vehicle.occupied):
                    entries = lane_maps[next_road][sublane]
                    if entries:
                        other_position, _, other = entries[0]
                        distance = (self.roads[road]["length"] - position +
                                    other_position)
                        if distance < front_distance:
                            front_distance = distance
                            leader = other
                            front = distance - (vehicle.length + other.length) / 2
        return leader, front, follower, back

    def _idm(self, vehicle, gap, leader_speed):
        if gap == math.inf:
            cached = self._free_idm_cache.get(id(vehicle))
            if cached is not None:
                return cached
        road_limit = (40.0 if vehicle.approach_end <= vehicle.position <
                      vehicle.connector_end else 60.0) / 3.6
        limit = min(vehicle.desired, road_limit)
        if vehicle.position < vehicle.approach_end:
            green = (vehicle.arm in ("NE", "SW") and self.phase == "LVL") or (
                vehicle.arm in ("SE", "NW") and self.phase == "HDT")
            if not green:
                stop_gap = vehicle.approach_end - vehicle.position + vehicle.gap
                if stop_gap < gap:
                    gap, leader_speed = stop_gap, 0.0
        elif vehicle.position > vehicle.connector_end:
            stop_gap = vehicle.route_end - vehicle.position + vehicle.gap
            if stop_gap < gap:
                gap, leader_speed = stop_gap, 0.0
        desired_gap = vehicle.gap + max(
            0.0, vehicle.speed * vehicle.headway +
            vehicle.speed * (vehicle.speed - leader_speed) /
            (2 * math.sqrt(vehicle.accel * vehicle.decel)))
        interaction = (desired_gap / gap) ** 2 if gap > 0 else math.inf
        result = vehicle.accel * (1 - (vehicle.speed / max(limit, 1e-6)) ** 4 - interaction)
        if gap == math.inf:
            self._free_idm_cache[id(vehicle)] = result
        return result

    def _measure(self):
        self.queue_time_integral += sum(v.speed * 3.6 < 1 for v in self.vehicles) * DT
        self.system_time_integral += len(self.vehicles) * DT
        self.weighted_delay_integral += sum(
            1 - min(1.0, v.speed / v.desired) for v in self.vehicles) * DT
        for arm in ARMS:
            used = set()
            for v in self.vehicles:
                if (v.arm == arm and (v.road == f"{arm}_in" or not v.road)
                        and 0 <= v.approach_end - v.position <= 1.5):
                    used.update(range(v.lane, v.lane + v.occupied))
            self.front_row_max[arm] = max(self.front_row_max[arm], len(used))

    def step(self, cycles=1):
        if not isinstance(cycles, int) or cycles < 0:
            raise ValueError("cycles must be a non-negative integer")
        for _ in range(cycles):
            self.cycle += 1
            self._advance_signal()
            self._spawn()
            self._measure()
            self._move()
        return self.snapshot()

    def snapshot(self):
        result = {"cycle": self.cycle, "time_sec": round(self.cycle * DT, 3),
                  "phase": self.phase, "created": self.created, "passed": self.passed,
                  "active": len(self.vehicles), "green_ticks": self.green_ticks,
                  "queue": sum(v.speed * 3.6 < 1 for v in self.vehicles),
                  "queue_time_integral": round(self.queue_time_integral, 3)}
        result.update(system_time_integral=round(self.system_time_integral, 3),
                      weighted_delay_integral=round(self.weighted_delay_integral, 3),
                      dqn_waiting_action=self.dqn_waiting_action,
                      next_decision_in_cycles=self.next_decision_in_cycles)
        result.update({f"front_row_max_{arm}": self.front_row_max[arm] for arm in ARMS})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=4000)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--flow-per-approach", type=float, default=2400)
    parser.add_argument("--motorbike-ratio", type=float, default=.85)
    parser.add_argument("--p-left", type=float, default=.2)
    parser.add_argument("--p-right", type=float, default=.2)
    parser.add_argument("--green-lvl", type=float, default=50)
    parser.add_argument("--green-hdt", type=float, default=40)
    parser.add_argument("--all-red", type=float, default=3)
    parser.add_argument("--demand-file", type=Path, default=None,
                        help="Replay demand CSV file (recommended for GAMA parity)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles < 0 or args.sample_every <= 0:
        parser.error("cycles must be non-negative and sample interval positive")
    if args.demand_file is None:
        sys.stderr.write("Warning: Running without --demand-file uses Python RNG for "
                         "vehicle generation; will not match GAMA replay demand.\n")
    sim = LeVanLuongSimulator(seed=args.seed,
        flow_per_approach=args.flow_per_approach,
        motorbike_ratio=args.motorbike_ratio, p_left=args.p_left, p_right=args.p_right,
        green_lvl=args.green_lvl, green_hdt=args.green_hdt, all_red=args.all_red,
        demand_file=args.demand_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sim.snapshot().keys())
        writer.writeheader()
        writer.writerow(sim.snapshot())
        for _ in range(args.cycles):
            row = sim.step()
            if sim.cycle % args.sample_every == 0:
                writer.writerow(row)
    print(json.dumps(sim.snapshot(), ensure_ascii=False))


if __name__ == "__main__":
    main()
