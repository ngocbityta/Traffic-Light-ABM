"""Independent, deterministic Python traffic model for the four-action learner.

This is a deliberately explicit approximation of GAMA's ``advanced_driving``
skill.  Demand, phase timing, observations, and rewards follow the GAML model;
longitudinal motion uses a lane-aware car-following model.  Validate aggregate
outputs against GAMA before using results for a scientific comparison.
"""

from __future__ import annotations

import csv
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gama_env import GamaEnvError, GamaTrafficEnv


DT = 0.1
STOP_DISTANCE = 949.5
OUT_LENGTH = 949.5
# Polyline lengths of the 121-point Bezier curves in init_doan_network.
CONNECTOR_LENGTH = {"straight": 81.0,
                    "left": 67.44206115652655,
                    "right": 123.38232154406313,
                    "uturn": 20.586709885884577}
BYPASS_LENGTHS = (1980.0, 1980.0, 1980.0, 1980.0,
                  1980.0, 1980.0, 1980.0, 1980.0)
ORIGIN_APPROACH = (5, 7, 4, 6)  # N, S, E, W
SOURCE_GROUP = (0, 1, 2, 3, 4, 5, 0, 1, 5, 4, 10, 11)
STRAIGHT_EXIT = (6, 7, 4, 5)   # E, N, W, S approach order
LEFT_EXIT = (7, 4, 5, 6)
RIGHT_EXIT = (5, 6, 7, 4)
PCU = (0.3, 1.0, 2.0)
LENGTH = (2.0, 4.6, 10.0)
WIDTH = (0.8, 1.9, 2.4)
ACCEL = (2.8, 1.8, 1.2)
DECEL = (4.5, 3.8, 2.8)
GAP = (0.8, 2.0, 3.0)
# Effective stop margin for the ramp convergence in this longitudinal model.
# GAML uses 70 m with a different per-agent reflex/IDM update order; paired
# vehicle traces place the Python stop several metres too early at 70 m.
FLYOVER_PORTAL_CLEARANCE = 66.0


@dataclass(slots=True)
class Vehicle:
    origin: int
    kind: int
    desired: float  # m/s
    maneuver: str
    lane: int
    approach: int
    destination: int
    length: float
    width: float
    spawn_cycle: int
    demand_row_id: int = -1
    position: float = 0.0  # metres along route
    speed: float = 0.0
    wait: float = 0.0
    lateral_peak: float = 0.0
    yellow_commit: bool = False
    left_committed: bool = False
    left_entered: bool = False
    last_lane_change_cycle: int = -100000
    ramp_stop_ticks: int = 0

    @property
    def ground(self):
        return self.origin < 4

    @property
    def signal(self):
        return self.ground and self.maneuver in ("straight", "left")

    @property
    def connector_start(self):
        return (910.0 if self.maneuver == "right" else
                920.0 if self.maneuver == "uturn" else STOP_DISTANCE)

    @property
    def connector_end(self):
        return self.connector_start + CONNECTOR_LENGTH[self.maneuver]

    @property
    def route_end(self):
        if self.ground:
            return self.connector_end + (910.0 if self.maneuver == "right" else
                                         920.0 if self.maneuver == "uturn" else OUT_LENGTH)
        return BYPASS_LENGTHS[self.origin - 4]


class PythonTrafficEnv(GamaTrafficEnv):
    """Async-compatible replacement for the four-phase GAMA server interface.

    The class keeps the public methods used by ``train.py`` and ``evaluate.py``.
    No Java process or WebSocket is started.  ``fidelity`` is intentionally
    exposed so a run cannot mistake this motion model for GAMA's driving skill.
    """

    fidelity = "approximate-longitudinal-v1"

    def __init__(self, source_clearance=15.0, speed_scale=1.0,
                 time_headway=1.5, controller_mode="dqn", **kwargs):
        super().__init__(**kwargs)
        if source_clearance <= 0 or speed_scale <= 0 or time_headway < 0:
            raise ValueError("Invalid source clearance, speed scale, or time headway")
        if controller_mode not in ("dqn", "fixed", "maxpressure"):
            raise ValueError(f"Unknown controller mode: {controller_mode}")
        self.source_clearance = float(source_clearance)
        self.speed_scale = float(speed_scale)
        self.time_headway = float(time_headway)
        self.controller_mode = controller_mode
        self.vehicles: list[Vehicle] = []
        self._rows = []
        self._pending = [deque() for _ in range(12)]
        self._cursor = 0
        self._rng = random.Random()
        self._seed = None
        self._model_demand = False
        self._params = {}
        self._action = None
        self._cycle = 0
        self._phase = 0
        self._phase_ticks = 0
        self._green_ticks = 0
        self._yellow_ticks = 0
        self._red_ticks = 0
        self._pending_phase = 0
        self._all_red = False
        self._waiting = False
        self._decisions = 0
        self._phase_wait = [0] * 4
        self._guard = False
        self._metrics = {}
        self._integrals = dict.fromkeys(("queue_time_integral", "external_wait_integral",
                                         "external_all_time_integral", "system_time_integral",
                                         "stopped_time_integral", "outgoing_block_integral",
                                         "weighted_delay_integral"), 0.0)
        self._count_created = 0
        self._count_passed = 0
        self._bypass_created = 0
        self._bypass_passed = 0
        self._travel_sum = 0.0
        self._sum_finished = 0
        self._flyover_portal_owner = {"NS": None, "SN": None}

    async def setup(self):
        return None

    async def close(self, stop_experiment=True):
        return None

    async def reset(self, seed=None, parameters=None):
        self._seed = int(seed if seed is not None else 1093)
        self._rng.seed(self._seed)
        self._params = {p["name"]: p["value"] for p in (parameters or [])}
        self._guard = str(self._params.get("dqn_service_guard", "false")).lower() == "true"
        demand_path = self._params.get("demand_file_path", "")
        self._model_demand = not bool(demand_path)
        self._rows = []
        if demand_path:
            with Path(demand_path).open(newline="", encoding="utf-8") as handle:
                for raw in csv.reader(handle):
                    if raw:
                        row = tuple(int(float(x)) for x in raw)
                        if len(row) != 6 or not 0 <= row[1] < 12 or not 0 <= row[2] <= 2:
                            raise ValueError(f"Invalid demand row: {raw}")
                        self._rows.append(row)
            if self._rows != sorted(self._rows, key=lambda r: (r[0], r[1])):
                raise ValueError("Demand CSV must be ordered by cycle and origin")
        self.vehicles = []
        self._pending = [deque() for _ in range(12)]
        self._cursor = 0
        self._action = None
        self._cycle = 0
        self._phase = self._phase_ticks = self._green_ticks = 0
        self._yellow_ticks = self._red_ticks = 0
        self._pending_phase = 0
        self._all_red = self._waiting = False
        self._decisions = 0
        self._phase_wait = [0] * 4
        self._count_created = self._count_passed = 0
        self._bypass_created = self._bypass_passed = 0
        self._travel_sum = 0.0
        self._sum_finished = 0
        self._flyover_portal_owner = {"NS": None, "SN": None}
        self._integrals = dict.fromkeys(("queue_time_integral", "external_wait_integral",
                                         "external_all_time_integral", "system_time_integral",
                                         "stopped_time_integral", "outgoing_block_integral",
                                         "weighted_delay_integral"), 0.0)
        self._metrics = {}
        self._measure(integrate=False)
        return await self.read_state()

    def _random_request(self, origin):
        kind = self._rng.choices((0, 1, 2), (.55, .38, .07))[0] if origin < 4 else 1
        speed = ((25 + self._rng.randint(0, 12)) if kind == 0 else
                 (30 + self._rng.randint(0, 15)) if kind == 1 else
                 (22 + self._rng.randint(0, 10)))
        maneuver = self._rng.choices((0, 1, 2, 3), (.60, .20, .15, .05))[0] if origin < 4 and kind != 2 else 0
        lane = self._rng.randint(4, 11) if kind == 0 else (8 if kind == 2 else self._rng.choice((4, 8)))
        return self._cycle, origin, kind, speed, maneuver, lane

    def _admit(self):
        if self._model_demand:
            if self._cycle % int(self._params.get("spawn_rate_NS", 15)) == 0:
                for origin in (0, 1):
                    self._pending[origin].append(self._random_request(origin))
            if self._cycle % int(self._params.get("spawn_rate_EW", 45)) == 0:
                for origin in (2, 3):
                    self._pending[origin].append(self._random_request(origin))
            interval = int(self._params.get("bypass_spawn_interval", 120))
            if self._cycle % interval == 0 and str(self._params.get("enable_bypass_traffic", "true")).lower() == "true":
                for origin in range(4, 12):
                    self._pending[origin].append(self._random_request(origin))
        else:
            while self._cursor < len(self._rows) and self._rows[self._cursor][0] <= self._cycle:
                row = self._rows[self._cursor]
                self._pending[row[1]].append((*row, self._cursor))
                self._cursor += 1
        interval = int(self._params.get("bypass_spawn_interval", 120))
        for k in range(12):
            origin = (k + self._cycle // interval) % 12
            if not self._pending[origin]:
                continue
            if any(SOURCE_GROUP[v.origin] == SOURCE_GROUP[origin] and
                   v.position < self.source_clearance for v in self.vehicles):
                continue
            # The GAML spawn guard measures distance in the plane to every
            # active ground vehicle.  An outgoing vehicle can therefore hold
            # the source on the opposite carriageway (four metres away).
            source_approach = (ORIGIN_APPROACH[origin] if origin < 4 else
                               5 if origin == 6 else 7 if origin == 7 else -1)
            if source_approach >= 0 and any(
                    (v.destination if v.ground else
                     5 if v.origin == 8 else 7 if v.origin == 9 else -1) ==
                    source_approach and
                    v.position >= (v.connector_end if v.ground else 1510.0) and
                    (v.route_end - v.position) ** 2 + 4.0 ** 2 <
                    self.source_clearance ** 2 for v in self.vehicles):
                continue
            request = self._pending[origin].popleft()
            _, _, kind, speed, turn, lane = request[:6]
            demand_row_id = request[6] if len(request) > 6 else -1
            maneuver = ("straight", "left", "right", "uturn")[turn] if origin < 4 else "straight"
            approach = ORIGIN_APPROACH[origin] if origin < 4 else -1
            idx = approach - 4
            destination = ((STRAIGHT_EXIT if maneuver == "straight" else
                            LEFT_EXIT if maneuver == "left" else
                            RIGHT_EXIT if maneuver == "right" else tuple(range(4, 8)))[idx]
                           if origin < 4 else -1)
            if origin < 4:
                lane = 12 if maneuver == "left" else 14 if maneuver == "uturn" else 0 if maneuver == "right" else lane
            else:
                lane = 14 if origin >= 10 else 9
            self.vehicles.append(Vehicle(origin, kind, speed / 3.6, maneuver, lane,
                                         approach, destination, LENGTH[kind], WIDTH[kind],
                                         self._cycle, demand_row_id))
            if origin < 4:
                self._count_created += 1
            else:
                self._bypass_created += 1

    def _junction_clear(self):
        return not any(v.signal and (
            STOP_DISTANCE <= v.position < v.connector_end + 5.0 + v.length / 2 or
            (v.yellow_commit and v.position < STOP_DISTANCE and STOP_DISTANCE - v.position <=
             4.5 + v.length / 2 + 2.0)) for v in self.vehicles)

    def _arbitrate_left_turns(self):
        """Admit one protected left turn per approach near the stop node."""
        if self._phase not in (2, 6):
            return
        for approach in ((5, 7) if self._phase == 2 else (4, 6)):
            if any(v.left_committed and v.approach == approach and
                   (not v.left_entered or v.position < v.connector_start + v.length + 12)
                   for v in self.vehicles):
                continue
            candidates = (v for v in self.vehicles if
                          v.approach == approach and v.maneuver == "left" and
                          not v.left_entered and not v.left_committed and
                          STOP_DISTANCE - 15 < v.position < STOP_DISTANCE and
                          not any(other is not v and other.ground and
                                  other.destination == v.destination and
                                  other.connector_end <= other.position <
                                  other.connector_end + v.length + 8
                                  for other in self.vehicles))
            nearest = max(candidates, key=lambda v: v.position, default=None)
            if nearest is not None:
                nearest.left_committed = True

    def _advance_controller(self):
        if self.controller_mode == "fixed":
            self._advance_fixed_controller()
            return
        if self.controller_mode == "maxpressure":
            self._advance_maxpressure_controller()
            return
        self._phase_ticks += 1
        demand = [0, 0, 0, 0]
        for v in self.vehicles:
            if v.signal and 0 <= STOP_DISTANCE - v.position <= 150:
                demand[(0 if v.approach in (5, 7) else 2) + (v.maneuver == "left")] += 1
        for a in range(4):
            if demand[a] == 0 or (self._phase == a * 2 and not self._all_red and self._phase % 2 == 0):
                self._phase_wait[a] = 0
            else:
                self._phase_wait[a] += 1
        if self._all_red:
            self._red_ticks += 1
            if self._red_ticks >= 20 and self._junction_clear():
                self._all_red = False
                self._red_ticks = 0
                self._phase = self._pending_phase
                self._phase_ticks = self._green_ticks = 0
        elif self._phase % 2:
            self._yellow_ticks += 1
            if self._yellow_ticks >= 30:
                self._yellow_ticks = 0
                self._all_red = True
        else:
            self._green_ticks += 1
            if self._waiting and self._action is not None:
                self._pending_phase = self._action * 2
                self._decisions += 1
                self._waiting = False
                self._action = None
                if self._pending_phase != self._phase:
                    self._phase += 1
                    self._yellow_ticks = 0
                    self._phase_ticks = 0
            elif not self._waiting and self._green_ticks >= 100 and self._green_ticks % 100 == 0:
                self._waiting = True

    def _advance_fixed_controller(self):
        durations = (300, 30, 120, 30, 300, 30, 120, 30)
        self._phase_ticks += 1
        required = durations[self._phase] + (20 if self._phase % 2 else 0)
        if self._phase_ticks >= required and (self._phase % 2 == 0 or self._junction_clear()):
            self._phase = (self._phase + 1) % 8
            self._phase_ticks = 0
        self._yellow_ticks = self._phase_ticks if self._phase % 2 else 0
        self._green_ticks = self._phase_ticks if self._phase % 2 == 0 else 0

    def _advance_maxpressure_controller(self):
        self._phase_ticks += 1
        if self._phase % 2:
            if self._phase_ticks >= 50 and self._junction_clear():
                self._phase = (self._phase + 1) % 8
                self._phase_ticks = 0
        elif self._phase in (2, 6):
            if self._phase_ticks >= 120:
                self._phase += 1
                self._phase_ticks = 0
        elif self._phase_ticks >= 100 and (self._phase_ticks % 50 == 0 or self._phase_ticks >= 600):
            incoming_ns = incoming_ew = outgoing_ns = outgoing_ew = waiting_left = 0
            for v in self.vehicles:
                if not v.ground:
                    continue
                if 0 <= STOP_DISTANCE - v.position <= 150:
                    if v.maneuver == "left" and v.speed * 3.6 < 1:
                        waiting_left += 1
                    elif v.maneuver == "straight":
                        if v.approach in (5, 7):
                            incoming_ns += 1
                        else:
                            incoming_ew += 1
                elif v.position >= v.connector_end and v.position < v.connector_end + 150 and 4 <= v.lane < 12:
                    if v.destination in (5, 7):
                        outgoing_ns += 1
                    else:
                        outgoing_ew += 1
            ns_pressure = incoming_ns - outgoing_ns
            ew_pressure = incoming_ew - outgoing_ew
            if ((self._phase == 0 and
                 (self._phase_ticks >= 600 or ew_pressure > ns_pressure + 2 or
                  (incoming_ns == 0 and (incoming_ew > 0 or waiting_left > 0)))) or
                (self._phase == 4 and
                 (self._phase_ticks >= 600 or ns_pressure > ew_pressure + 2 or
                  (incoming_ew == 0 and (incoming_ns > 0 or waiting_left > 0))))):
                self._phase += 1
                self._phase_ticks = 0
        self._yellow_ticks = self._phase_ticks if self._phase % 2 else 0
        self._green_ticks = self._phase_ticks if self._phase % 2 == 0 else 0

    def _leader_gaps(self):
        """Nearest overlapping-lane leader on each physical road edge."""
        groups = {}
        for v in self.vehicles:
            if v.ground:
                if v.position < v.connector_start:
                    section, progress = ("incoming",) + self._incoming_edge(v), v.position
                elif v.position < v.connector_end:
                    section, progress = ("connector", v.origin, v.maneuver), v.position
                else:
                    section, progress = ("outgoing", v.destination), v.position - v.connector_end
            elif v.origin in (4, 6, 9):
                # flyover_NS and off_S share the approach; on_N joins at 630 m.
                # off_S separates at 1,350 m.
                if v.origin == 6 and v.position < 470:
                    section = ("incoming", 0, 0)
                elif v.position < 630 and v.origin == 6:
                    section = ("on_N",)
                elif v.position < 630:
                    section = ("F_NS0",)
                elif v.position < 1350:
                    section = ("F_NS1",)
                elif v.origin == 9 and v.position >= 1510:
                    section = ("outgoing", 7)
                elif v.origin == 9:
                    section = ("off_S",)
                else:
                    section = ("F_NS2",)
                progress = (479.5 + v.position - 1510.0
                            if v.origin == 9 and v.position >= 1510 else v.position)
            elif v.origin in (5, 7, 8):
                if v.origin == 7 and v.position < 470:
                    section = ("incoming", 1, 0)
                elif v.position < 630 and v.origin == 7:
                    section = ("on_S",)
                elif v.position < 630:
                    section = ("F_SN0",)
                elif v.position < 1350:
                    section = ("F_SN1",)
                elif v.origin == 8 and v.position >= 1510:
                    section = ("outgoing", 5)
                elif v.origin == 8:
                    section = ("off_N",)
                else:
                    section = ("F_SN2",)
                progress = (479.5 + v.position - 1510.0
                            if v.origin == 8 and v.position >= 1510 else v.position)
            else:
                section, progress = ("bypass", v.origin), v.position
            groups.setdefault(section, []).append((v, progress))
        gaps = {}
        for group in groups.values():
            group.sort(key=lambda item: item[1], reverse=True)
            for i, (follower, follower_progress) in enumerate(group):
                gap = math.inf
                leader_speed = 0.0
                for leader, leader_progress in reversed(group[:i]):
                    if follower.lane < leader.lane + (1 if leader.kind == 0 else 2) and leader.lane < follower.lane + (1 if follower.kind == 0 else 2):
                        gap = max(0.0, leader_progress - follower_progress -
                                  (leader.length + follower.length) / 2)
                        leader_speed = leader.speed
                        break
                gaps[id(follower)] = (gap, leader_speed)
        return gaps

    def _reserve_flyover_portals(self):
        """Reserve the two ramp/flyover convergence nodes before driving."""
        for direction, through, ramp, off in (("NS", 4, 6, 9),
                                               ("SN", 5, 7, 8)):
            owner = self._flyover_portal_owner[direction]
            if owner is not None and (not any(v is owner for v in self.vehicles) or
                                      owner.position > 630.0 + owner.length / 2.0 + 3.0):
                owner = None
            candidates = [v for v in self.vehicles if v.origin in (through, ramp, off) and
                          510.0 < v.position < 630.0]
            for v in candidates:
                if v.origin == ramp and v is not owner and v.speed * 3.6 < 0.5:
                    v.ramp_stop_ticks += 1
                else:
                    v.ramp_stop_ticks = 0
            if owner is None:
                ramp_candidates = [v for v in candidates if v.origin == ramp]
                ramp_head = max(ramp_candidates, key=lambda v: v.position,
                                default=None)
                if ramp_head is not None and ramp_head.ramp_stop_ticks >= 50:
                    owner = ramp_head
                elif candidates:
                    owner = max(candidates, key=lambda v: v.position)
            self._flyover_portal_owner[direction] = owner

    def _incoming_edge(self, vehicle):
        if not vehicle.ground or vehicle.position >= vehicle.connector_start:
            return None
        portal = 470.0 if vehicle.origin < 2 else 830.0
        position = vehicle.position
        edge = 0 if position < portal else 1 if position < 910.0 else 2 if position < 920.0 else 3
        return vehicle.origin, edge

    @staticmethod
    def _overlap(a_lane, a_kind, b_lane, b_kind):
        return (a_lane < b_lane + (1 if b_kind == 0 else 2) and
                b_lane < a_lane + (1 if a_kind == 0 else 2))

    def _choose_lanes(self):
        """GAML's safe adjacent-lane gate followed by an IDM/MOBIL gain check."""
        groups = {}
        for vehicle in self.vehicles:
            edge = self._incoming_edge(vehicle)
            if edge is not None:
                groups.setdefault(edge, []).append(vehicle)
        gaps = self._leader_gaps()
        for vehicle in self.vehicles:
            edge = self._incoming_edge(vehicle)
            if (edge is None or vehicle.maneuver != "straight" or
                    vehicle.position >= STOP_DISTANCE - 13.0 or
                    self._cycle - vehicle.last_lane_change_cycle < 30):
                continue
            nearby = groups[edge]
            slowed = any(other is not vehicle and other.position > vehicle.position and
                          self._overlap(vehicle.lane, vehicle.kind, other.lane, other.kind) and
                          other.position - vehicle.position - (vehicle.length + other.length)/2 <
                          max(12.0, vehicle.speed * 2.0) and
                          other.speed + 1.0 < vehicle.speed for other in nearby)
            if not slowed:
                continue
            allowed = (range(4, 12) if vehicle.kind == 0 else
                       (8,) if vehicle.kind == 2 else range(4, 11))
            speed_limit = min(vehicle.desired, 50.0 / 3.6)
            current_gap, current_leader_speed = gaps[id(vehicle)]
            current_accel = self._idm_acceleration(vehicle, speed_limit,
                                                    current_gap, current_leader_speed)
            for candidate in allowed:
                if abs(candidate - vehicle.lane) != 1:
                    continue
                safe = True
                target_gap = math.inf
                target_speed = 0.0
                target_follower = None
                follower_gap = math.inf
                for other in nearby:
                    if other is vehicle or not self._overlap(candidate, vehicle.kind,
                                                              other.lane, other.kind):
                        continue
                    along = other.position - vehicle.position
                    bumper_gap = abs(along) - (vehicle.length + other.length)/2
                    if ((along >= 0 and bumper_gap < max(4.0, vehicle.speed * 1.5)) or
                            (along < 0 and bumper_gap <
                             max(5.0, (other.speed - vehicle.speed) * 2.5 + 5.0))):
                        safe = False
                        break
                    if along > 0 and bumper_gap < target_gap:
                        target_gap, target_speed = bumper_gap, other.speed
                    elif along < 0 and bumper_gap < follower_gap:
                        target_follower, follower_gap = other, bumper_gap
                if not safe:
                    continue
                target_accel = self._idm_acceleration(vehicle, speed_limit,
                                                      target_gap, target_speed)
                follower_delta = 0.0
                if target_follower is not None:
                    follower_limit = min(target_follower.desired, 50.0 / 3.6)
                    after = self._idm_acceleration(target_follower, follower_limit,
                                                   follower_gap, vehicle.speed)
                    if after <= -3.0:
                        continue
                    prior_gap = math.inf
                    prior_speed = 0.0
                    for other in nearby:
                        if (other is vehicle or other is target_follower or
                                other.position <= target_follower.position or
                                not self._overlap(target_follower.lane, target_follower.kind,
                                                  other.lane, other.kind)):
                            continue
                        other_gap = (other.position - target_follower.position -
                                     (other.length + target_follower.length)/2)
                        if other_gap < prior_gap:
                            prior_gap, prior_speed = other_gap, other.speed
                    before = self._idm_acceleration(target_follower, follower_limit,
                                                    prior_gap, prior_speed)
                    follower_delta = after - before
                if target_accel - current_accel + 0.5 * follower_delta <= 0.5:
                    break
                old_lane = vehicle.lane
                vehicle.lane = candidate
                gaps = self._leader_gaps()
                vehicle.last_lane_change_cycle = self._cycle
                # Smooth lateral movement peaks at roughly 2.8 m/s for a
                # one-sublane car change.  GAML scales duration by type.
                multiplier = (0.6 if vehicle.kind == 0 else
                              1.8 if vehicle.kind == 2 else 1.0)
                peak = (old_lane - candidate) * (2.8 / multiplier)
                if abs(peak) > abs(vehicle.lateral_peak):
                    vehicle.lateral_peak = peak
                break

    def _idm_acceleration(self, vehicle, speed_limit, gap, leader_speed):
        """GAMA 2025.6.4 IDM.computeAcceleration, with metres and seconds."""
        speed = vehicle.speed
        desired_gap = GAP[vehicle.kind] + max(
            0.0, speed * self.time_headway +
            speed * (speed - leader_speed) /
            (2.0 * math.sqrt(ACCEL[vehicle.kind] * DECEL[vehicle.kind])))
        interaction = (desired_gap / gap) ** 2 if gap > 0 else math.inf
        return ACCEL[vehicle.kind] * (1.0 - (speed / max(speed_limit, 1e-6)) ** 4 - interaction)

    def _advance_vehicles(self):
        gaps = self._leader_gaps()
        completed = []
        for v in self.vehicles:
            # GAML count_waiting runs before the vehicle's driving reflex.
            if v.ground and v.speed * 3.6 < 1.0:
                v.wait += DT
            speed_limit = v.desired * self.speed_scale
            if v.ground and v.connector_start <= v.position < v.connector_end:
                speed_limit = min(speed_limit, (18 if v.maneuver == "left" else
                                                15 if v.maneuver == "right" else
                                                8 if v.maneuver == "uturn" else 50) / 3.6)
            elif v.ground and v.position > v.connector_end:
                speed_limit = min(speed_limit, 50 / 3.6)
            elif not v.ground and v.origin in (6, 7, 8, 9):
                ramp_start = 470.0 if v.origin in (6, 7) else 1350.0
                if ramp_start <= v.position < ramp_start + 160.0:
                    speed_limit = min(speed_limit, 20.0 / 3.6)
                elif v.position < ramp_start:
                    usable = max(0.0, ramp_start - v.position -
                                 v.length / 2.0 - GAP[v.kind] - 5.0)
                    speed_limit = min(speed_limit,
                        math.sqrt((20.0 / 3.6) ** 2 +
                                  2.0 * DECEL[v.kind] * usable))
            portal_direction = ("NS" if v.origin in (4, 6, 9) else
                                "SN" if v.origin in (5, 7, 8) else None)
            portal_wait = (portal_direction is not None and
                           v.position < 630.0 and
                           self._flyover_portal_owner[portal_direction] is not v)
            if portal_wait:
                merge_usable = (630.0 - v.position - FLYOVER_PORTAL_CLEARANCE -
                                v.length / 2.0 - GAP[v.kind] - 2.0)
                if merge_usable <= 0.0 and v.speed * 3.6 < 1.5:
                    v.speed = 0.0
                    continue
                usable = max(0.0, 630.0 - v.position - FLYOVER_PORTAL_CLEARANCE -
                             v.length / 2.0 - GAP[v.kind] - 4.0)
                cap = max(0.01 / 3.6, math.sqrt(2.0 * DECEL[v.kind] * usable))
                if usable <= 0.0:
                    cap = max(0.01 / 3.6, v.speed - DECEL[v.kind] * DT)
                else:
                    cap = max(cap, v.speed - DECEL[v.kind] * DT,
                              v.speed / (1.0 + DECEL[v.kind] / ACCEL[v.kind]) ** 0.25)
                speed_limit = min(speed_limit, cap)
            if v.maneuver == "left" and not v.left_entered and v.position < STOP_DISTANCE:
                remaining = max(0.0, STOP_DISTANCE - v.position)
                if not v.left_committed:
                    usable = max(0.0, remaining - 4.5 - v.length / 2 - 2.0)
                    speed_limit = min(speed_limit, max(0.1 / 3.6,
                        min(math.sqrt(0.2 * DECEL[v.kind] * usable), usable / 4.0)))
                if remaining < 55.0:
                    speed_limit = min(speed_limit,
                        math.sqrt((18.0 / 3.6) ** 2 + 2.0 * DECEL[v.kind] * remaining))
            gap, leader_speed = gaps[id(v)]
            # DrivingSkill's findLeader returns the final target as a stopped
            # virtual leader at distance_to_current_target + min_safety_distance.
            # This makes cars slow down before reaching the terminal node and
            # keeps them in the spawn exclusion zone for several more ticks.
            if v.position >= v.connector_end or not v.ground:
                target_gap = max(0.0, v.route_end - v.position) + GAP[v.kind]
                if target_gap < gap:
                    gap, leader_speed = target_gap, 0.0
            stop_gap = math.inf
            if v.signal and v.position < STOP_DISTANCE:
                served = (0 if v.approach in (5, 7) else 2) + (v.maneuver == "left")
                green = (self._phase == 2 * served and not self._all_red and
                         (v.maneuver != "left" or v.left_committed))
                yellow = (self._phase == 2 * served + 1 and not self._all_red and
                          self._yellow_ticks < 30 and
                          (v.maneuver != "left" or v.left_committed or v.left_entered))
                stop_gap = STOP_DISTANCE - v.position - 4.5 - v.length / 2 - 2.0
                if yellow and stop_gap < v.speed * v.speed / (2 * DECEL[v.kind]) + 6.0:
                    v.yellow_commit = True
                elif not yellow and stop_gap > 4.5 + v.length / 2 + 2.0:
                    v.yellow_commit = False
                if not green and not (v.yellow_commit and
                                      (yellow or stop_gap <= 4.5 + v.length / 2 + 2.0)):
                    usable = max(0.0, stop_gap)
                    safe_cap = min(math.sqrt(0.2 * DECEL[v.kind] * usable), usable / 4.0)
                    commanded_cap = max(0.1 / 3.6, safe_cap)
                    if self._phase % 2 or self._phase_ticks < 10:
                        factor = 3.0 if yellow else 2.0
                        commanded_cap = max(commanded_cap, v.speed - factor * DECEL[v.kind] * DT)
                    final_edge = v.position >= STOP_DISTANCE - 29.5
                    factor = (2.5 if v.speed > 2.0 / 3.6 else 1.0) if final_edge else 1.0
                    speed_limit = min(speed_limit,
                                      max(commanded_cap, v.speed - factor * DECEL[v.kind] * DT))
                    if stop_gap <= max(0.5, v.speed * DT) and (
                            v.speed < 2.0 / 3.6 or stop_gap <= 0.0):
                        v.speed = 0.0
                        continue
            acceleration = self._idm_acceleration(v, speed_limit, gap, leader_speed)
            if not math.isfinite(acceleration):
                acceleration = -1e12
            if v.speed + acceleration * DT < 0.0:
                distance = -0.5 * v.speed * v.speed / acceleration
                speed = 0.0
            else:
                distance = v.speed * DT + 0.5 * acceleration * DT * DT
                speed = v.speed + acceleration * DT
            if math.isfinite(gap):
                distance = min(distance, max(0.0, gap))
                speed = min(speed, max(0.0, gap / DT))
            if math.isfinite(stop_gap) and not green and not v.yellow_commit:
                distance = min(distance, max(0.0, stop_gap))
            if portal_wait:
                distance = min(distance, max(0.0, 630.0 - v.position - 0.01))
                speed = min(speed, max(0.0, (630.0 - v.position - 0.01) / DT))
            v.speed = speed
            v.position += max(0.0, distance)
            if v.maneuver == "left" and v.position >= STOP_DISTANCE:
                v.left_entered = True
            # moveAcrossSegments snaps to a node when its final segment is
            # shorter than 0.01 m; without this, IDM approaches the goal
            # asymptotically and the Python vehicle never completes.
            if v.position >= v.route_end - 0.01:
                completed.append(v)
        if completed:
            done = {id(v) for v in completed}
            self.vehicles = [v for v in self.vehicles if id(v) not in done]
            for v in completed:
                if v.ground:
                    self._count_passed += 1
                    self._travel_sum += (self._cycle - v.spawn_cycle) * DT
                    self._sum_finished += 1
                else:
                    self._bypass_passed += 1

    def _measure(self, integrate=True):
        ground = [v for v in self.vehicles if v.ground]
        ground_pending = sum(len(q) for q in self._pending[:4])
        bypass_pending = sum(len(q) for q in self._pending[4:])
        stopped = sum(v.speed * 3.6 < 1.0 for v in ground)
        all_stopped = sum(v.speed * 3.6 < 1.0 for v in self.vehicles)
        blocked = sum(v.ground and v.position >= v.connector_end and
                      v.position < v.connector_end + 80 and v.speed * 3.6 < 5
                      for v in self.vehicles)
        weighted = sum(PCU[v.kind] * (1 - min(1.0, v.speed / max(.1 / 3.6, v.desired)))
                       for v in self.vehicles)
        weighted += sum(PCU[row[2]] for queue in self._pending for row in queue)
        waits = [v.wait for v in ground]
        self._metrics = {
            "current_queue_length": stopped,
            "average_waiting_time": sum(waits) / len(waits) if waits else 0.0,
            "max_waiting_time": max(waits, default=0.0),
            "average_speed": sum(v.speed * 3.6 for v in ground) / len(ground) if ground else 0.0,
            "average_travel_time": self._travel_sum / self._sum_finished if self._sum_finished else 0.0,
        }
        if integrate:
            updates = {"queue_time_integral": stopped, "external_wait_integral": ground_pending,
                       "external_all_time_integral": ground_pending + bypass_pending,
                       "system_time_integral": len(self.vehicles) + ground_pending + bypass_pending,
                       "stopped_time_integral": all_stopped,
                       "outgoing_block_integral": blocked,
                       "weighted_delay_integral": weighted}
            for name, value in updates.items():
                self._integrals[name] += value * DT

    async def step_cycles(self, n):
        if int(n) != n or n < 0:
            raise ValueError(f"Cycle count must be non-negative integer: {n}")
        for _ in range(int(n)):
            self._cycle += 1
            self._advance_controller()
            self._arbitrate_left_turns()
            self._admit()
            self._choose_lanes()
            self._reserve_flyover_portals()
            self._advance_vehicles()
            self._measure()

    async def set_action(self, action):
        if self.controller_mode != "dqn" or action not in range(4) or not self._waiting:
            raise GamaEnvError(f"Action {action} outside a decision tick")
        self._action = int(action)

    async def get_next_decision_in(self):
        if self.controller_mode != "dqn":
            return 1
        return (1 if self._waiting else
                max(1, 20 - self._red_ticks + 100) if self._all_red else
                30 - self._yellow_ticks + 20 + 100 if self._phase % 2 else
                100 - self._green_ticks % 100)

    def _observations(self, state):
        demand = [0] * 4
        for v in self.vehicles:
            if v.signal and 0 <= STOP_DISTANCE - v.position <= 150:
                demand[(0 if v.approach in (5, 7) else 2) + (v.maneuver == "left")] += 1
        state["queue_ns"] = sum(v.signal and v.approach in (5, 7) and
                                0 <= STOP_DISTANCE - v.position <= 150 and v.speed * 3.6 < 1
                                for v in self.vehicles)
        state["queue_ew"] = sum(v.signal and v.approach in (4, 6) and
                                0 <= STOP_DISTANCE - v.position <= 150 and v.speed * 3.6 < 1
                                for v in self.vehicles)
        state["approach_ns"] = demand[0] + demand[1] - state["queue_ns"]
        state["approach_ew"] = demand[2] + demand[3] - state["queue_ew"]
        mask = np.ones(4, dtype=np.bool_)
        if self._guard:
            old = [i for i in range(4) if self._phase_wait[i] >= 1200]
            if old:
                mask[:] = False
                mask[max(old, key=lambda i: self._phase_wait[i])] = True
            elif self._green_ticks >= 600 and self._phase % 2 == 0:
                current = self._phase // 2
                if sum(demand) > demand[current]:
                    mask[current] = False
        state["action_mask"] = mask
        if self.state_mode == "dtse":
            cells = np.zeros((3, 8, 20), dtype=np.float32)
            counts = np.zeros((8, 20), dtype=np.int32)
            delay_all = delay_near = 0.0
            for v in self.vehicles:
                if not v.signal or v.position >= STOP_DISTANCE or v.approach not in range(4, 8):
                    continue
                fraction = min(1.0, v.speed * 3.6 / 50)
                d = 990.0 - v.position - 40.5
                delay = 1.0 - fraction * fraction
                delay_all += delay
                if 0 <= d < 160:
                    row, col = (v.approach - 4) * 2 + (v.maneuver == "left"), int(d / 8)
                    cells[0, row, col] = 1
                    cells[1, row, col] += fraction
                    counts[row, col] += 1
                    delay_near += delay
            np.divide(cells[1], counts, out=cells[1], where=counts > 0)
            for row in range(8):
                phase = (0 if row // 2 in (1, 3) else 2) + row % 2
                if self._phase == phase * 2 and not self._all_red:
                    cells[2, row, :] = 1
            state["dtse"] = cells
            state["delay_sq"] = delay_all if self.reward_scope == "all" else delay_near
            state["queue_time_integral"] = self._integrals["queue_time_integral"]
            state["external_wait_integral"] = self._integrals["external_wait_integral"]
        else:
            grid = np.zeros((4, 6, 20, 16), dtype=np.float32)
            grid[:, 5] = 1.0
            area = np.zeros((4, 20, 16), dtype=np.float32)
            incoming = np.zeros(8, dtype=np.float32)
            outgoing = np.zeros(4, dtype=np.float32)
            for v in self.vehicles:
                if v.ground and v.position >= v.connector_end and v.position < v.connector_end + 80 and v.speed * 3.6 < 5:
                    outgoing[v.destination - 4] += 1
                if not v.signal or v.position >= STOP_DISTANCE or v.approach not in range(4, 8):
                    continue
                arm = v.approach - 4
                along = STOP_DISTANCE - v.position - 4.5
                if 0 <= along < 160 and v.speed * 3.6 < 5:
                    incoming[arm * 2 + (v.maneuver == "left")] += 1
                lo, hi = along - v.length / 2, along + v.length / 2
                lateral = v.lane + (1 if v.kind == 0 else 2) / 2
                wlo, whi = lateral - v.width / 3.5, lateral + v.width / 3.5
                for h in range(max(0, int(math.floor(lo / 8))), min(19, int(math.floor(hi / 8))) + 1):
                    overlap_h = max(0, min(hi, (h + 1) * 8) - max(lo, h * 8)) / 8
                    for w in range(max(0, int(math.floor(wlo))), min(15, int(math.floor(whi))) + 1):
                        a = overlap_h * max(0, min(whi, w + 1) - max(wlo, w))
                        grid[arm, v.kind, h, w] = min(1, grid[arm, v.kind, h, w] + a)
                        grid[arm, 3, h, w] += a * min(1, v.speed * 3.6 / 50)
                        grid[arm, 4, h, w] += a * max(-1.0, min(1.0,
                            -v.lateral_peak / 5.0))
                        area[arm, h, w] += a
            np.divide(grid[:, 3], area, out=grid[:, 3], where=area > 0)
            np.divide(grid[:, 4], area, out=grid[:, 4], where=area > 0)
            state["spatial"] = grid
            state["pressure"] = np.asarray([
                incoming[2] - outgoing[3] + incoming[6] - outgoing[1],
                incoming[3] - outgoing[0] + incoming[7] - outgoing[2],
                incoming[0] - outgoing[2] + incoming[4] - outgoing[0],
                incoming[1] - outgoing[3] + incoming[5] - outgoing[1]], dtype=np.float32)
            state.update(self._integrals)
            for v in self.vehicles:
                v.lateral_peak = 0.0

    async def read_state(self, force_dtse=False):
        state = dict(self._metrics)
        for name in ("average_waiting_time", "max_waiting_time", "average_speed",
                     "average_travel_time"):
            state[name] = round(state[name], 2)
        state.update(cycle=self._cycle, global_phase=self._phase,
                     green_cycles=self._green_ticks, yellow_cycles=self._yellow_ticks,
                     total_vehicles_passed=self._count_passed,
                     total_vehicles_created=self._count_created,
                     decision_count=self._decisions,
                     dqn_waiting_action=self._waiting,
                     next_decision_in_cycles=await self.get_next_decision_in())
        if self._waiting or force_dtse:
            self._observations(state)
        return state
