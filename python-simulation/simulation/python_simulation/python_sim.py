#!/usr/bin/env python3
"""GeoJSON-backed Python simulator for the Le Van Luong GAML layout.

Road lengths, sublane counts, turns, signal cycle and arrival probabilities
come directly from the current GeoJSON and LeVanLuong.gaml.  Vehicle motion
uses ported IDM and MOBIL rules; paired benchmarks still quantify remaining
individual-vehicle differences.

`LeVanLuongSimulator(mode="safe")` preserves the simulator's additional road
and spawn safety rules. Set `mode="gama_2025_6_4"` to use the installed GAMA
2025.6.4 IDM/MOBIL, startup registration and node transition behavior for
paired demand replay. Safe mode remains the default. Unpaired random arrivals
use Python's RNG and are not a deterministic GAMA spawn replay.
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

if __package__:
    from python_simulation.gama_2025_idm_mobil import (
        Candidate as GamaCandidate, IDMParams as GamaIDMParams,
        MobilParams as GamaMobilParams, choose_lane as gama_choose_lane,
        idm_acceleration as gama_idm_acceleration,
    )
    from python_simulation.gama_2025_transition import transition_at_node
    from python_simulation.safe_startup import (
        SafeStartupScheduler, StartupVehicle,
    )
else:  # Support `python python-simulation/simulation/python_simulation/python_sim.py` and legacy imports.
    from gama_2025_idm_mobil import (
        Candidate as GamaCandidate, IDMParams as GamaIDMParams,
        MobilParams as GamaMobilParams, choose_lane as gama_choose_lane,
        idm_acceleration as gama_idm_acceleration,
    )
    from gama_2025_transition import transition_at_node
    from safe_startup import SafeStartupScheduler, StartupVehicle

try:
    from .vehicle_geometry import Road as GeometryRoad, SUBLANE_WIDTH
    from .junction_conflicts import (ConflictEngine, ConflictMap, Movement,
                                    MovementRequest)
    from .junction_negotiation import SimulatorAdapter
    from .vehicle_geometry import footprint as body_polygon, polygons_overlap
except ImportError:
    from vehicle_geometry import Road as GeometryRoad, SUBLANE_WIDTH
    from junction_conflicts import ConflictEngine, ConflictMap, Movement, MovementRequest
    from junction_negotiation import SimulatorAdapter
    from vehicle_geometry import footprint as body_polygon, polygons_overlap


ROOT = Path(__file__).resolve().parent
ARMS = ("NE", "SE", "SW", "NW")
DT = 0.2
STOP_MARGIN = 0.5
LATERAL_CLEARANCE = 0.1
# A stationary entry grant older than this is withdrawn and re-requested.
TOKEN_STALL_LIMIT = 2.0
TOKEN_STALL_COOLDOWN = 1.0
# Seconds a vehicle waiting to merge into its route corridor needs before it
# accepts the minimum physical gap (forced/courtesy merge, cf. SUMO
# lcAssertive/impatience). Behavioural assumption, not calibrated.
MERGE_PATIENCE = 8.0


def turn_lane_bounds(lane_count, occupied, maneuver):
    """Turn corridors and the required through queue corridor, from the right curb."""
    last = lane_count - occupied
    right_last = max(0, lane_count // 4 - occupied)
    left_first = max(0, last - 2)
    if maneuver == "right":
        return 0, min(last, right_last)
    if maneuver == "left":
        return left_first, last
    # A through vehicle must not queue in either exclusive turn corridor.
    first = min(last, lane_count // 4)
    final = max(first, left_first - occupied)
    return first, final


class _RoadIndex:
    """Vehicles on one road sorted by road progress (one entry per vehicle).

    Maintained automatically by ``_Sublane`` inserts/pops, so every existing
    ``bisect.insort``/``pop`` site in ``_move`` keeps it current. It is used
    only as a candidate filter in front of the original pairwise predicates;
    a ``margin`` keeps the candidate set a strict superset.
    """
    __slots__ = ("entries", "counts")

    def __init__(self):
        self.entries = []
        self.counts = {}

    def add(self, entry):
        key = (entry[0], entry[1])
        count = self.counts.get(key, 0)
        if not count:
            bisect.insort(self.entries, entry)
        self.counts[key] = count + 1

    def discard(self, entry):
        key = (entry[0], entry[1])
        count = self.counts.get(key, 0)
        if count > 1:
            self.counts[key] = count - 1
            return
        self.counts.pop(key, None)
        index = bisect.bisect_left(self.entries, key)
        if (index < len(self.entries) and self.entries[index][0] == key[0]
                and self.entries[index][1] == key[1]):
            del self.entries[index]

    def window(self, low, high):
        """Vehicles whose indexed progress lies in ``[low, high]``."""
        entries = self.entries
        start = bisect.bisect_left(entries, (low, -1))
        out = []
        for index in range(start, len(entries)):
            if entries[index][0] > high:
                break
            out.append(entries[index][2])
        return out


class _Sublane(list):
    """Sorted sublane list that mirrors inserts/pops into its road index."""
    __slots__ = ("road_index",)

    def __init__(self, road_index):
        super().__init__()
        self.road_index = road_index

    def insert(self, index, entry):
        list.insert(self, index, entry)
        self.road_index.add(entry)

    def pop(self, index=-1):
        entry = list.pop(self, index)
        self.road_index.discard(entry)
        return entry


# Stop-or-go decision at yellow onset: binary logit, Point Data Model of
# Chauhan, Dhamaniya & Arkatkar (2022), Transportation Research Part F 89,
# 222-235, Table "Model coefficients ... Point Data Model". U_pass =
# 4.518 + 0.106 v[m/s] - 0.083 x[m] + vehicle class + lane position (base:
# motorised two-wheeler, kerb-side lane). Indian data (left-hand traffic,
# mixed 2W-dominant flow); transferability to Hanoi is an assumption.
CHAUHAN2022_PDM = {
    "constant": 4.518165, "speed": 0.106, "distance": -0.083,
    "car": -1.068, "median_lane": -1.049, "intermediate_lane": -0.923,
}
# Above this deceleration a "stop" is physically infeasible -> the vehicle goes.
# (Chauhan et al. report decelerations down to about -6 m/s^2 in the amber.)
STOP_GO_MAX_DECEL = 6.0
# Beyond this distance from the stop line at yellow the vehicle simply stops.
STOP_GO_MAX_DISTANCE = 150.0

# Car-following parameters per vehicle type (IDM: a, b, s0, T). Defaults are
# the model's original engineering values, not calibrated; override them with
# ``vehicle_params`` (see calibration.py).
DEFAULT_VEHICLE_PARAMS = {
    "car": {"accel": 2.5, "decel": 3.0, "gap": 1.0, "headway": 1.2},
    "motorbike": {"accel": 2.0, "decel": 3.0, "gap": 0.3, "headway": 0.6},
}

# Candidate-window safety margin (m) around indexed progress.
INDEX_MARGIN = 0.5


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
    creation_id: int = -1
    created_cycle: int = 0
    current_lateral_shift: float = 0.0
    start_lateral_shift: float = 0.0
    target_lateral_shift: float = 0.0
    lane_change_progress: float = 1.0
    lateral_duration: float = 0.8
    lane_change_from: int = -1
    gama_entered: bool = True
    seam_from_road: str = ""
    seam_to_road: str = ""
    seam_progress: float = 0.0
    seam_shift: float = 0.0
    junction_wait_since: float | None = None
    junction_entry_token: bool = False
    token_stall_time: float = 0.0
    token_cooldown_until: float = -1.0
    merge_wait: float = 0.0
    # Stop-or-go decision for the yellow interval ``yellow_epoch``.
    yellow_go: bool = False
    yellow_epoch: int = -1
    # Lateral shift of the corridor sublane an impatient merger is claiming;
    # followers in that sublane yield to it (courtesy gap). NaN = no claim.
    merge_claim_shift: float = math.nan
    queue_stop_offset: float = 0.0

    in_road: str = ""
    conn_road: str = ""
    out_road: str = ""

    def __post_init__(self):
        self.in_road = f"{self.arm}_in"
        self.conn_road = f"{self.arm}_{self.to_arm}"
        self.out_road = f"{self.to_arm}_out"

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
    """Le Van Luong simulation with safe (default) or GAMA 2025.6.4 mode."""
    @staticmethod
    def _half_width(vehicle):
        # Physical widths from the model: 1.7 m cars and 0.7 m motorbikes.
        return 0.85 if vehicle.occupied == 2 else 0.35

    def _footprint(self, vehicle):
        first, last = vehicle.lane, vehicle.lane + vehicle.occupied
        if vehicle.lane_change_progress < 1.0 and vehicle.lane_change_from >= 0:
            first = min(first, vehicle.lane_change_from)
            last = max(last, vehicle.lane_change_from + vehicle.occupied)
        return range(first, last)

    def _continuous_seam_pose(self, old_road, next_road, old_shift):
        """Return next-road progress and lateral shift at the old center point."""
        old, new = self._geometry_roads[old_road], self._geometry_roads[next_road]
        before = old.pose(old.length, old_shift)
        origin = new.pose(0.0, new.num_lanes * SUBLANE_WIDTH - SUBLANE_WIDTH / 2.0)
        nx, ny = origin.ty, -origin.tx
        dx, dy = before.x - origin.x, before.y - origin.y
        along = dx * origin.tx + dy * origin.ty
        offset = dx * nx + dy * ny
        shift = new.num_lanes * SUBLANE_WIDTH - SUBLANE_WIDTH / 2.0 - offset
        return along, shift

    def _route_target_lane(self, vehicle):
        if vehicle.road != f"{vehicle.arm}_in":
            return vehicle.lane
        incoming = self.roads[f"{vehicle.arm}_in"]
        minimum, maximum = turn_lane_bounds(
            incoming["num_lanes"], vehicle.occupied, vehicle.maneuver)
        return min(max(vehicle.lane, minimum), maximum)

    def _mapped_lane(self, vehicle, lane, old_road, next_road):
        old_max = self.roads[old_road]["num_lanes"] - vehicle.occupied
        next_max = self.roads[next_road]["num_lanes"] - vehicle.occupied
        if self.mode == "gama_2025_6_4":
            return min(lane, next_max)
        if vehicle.maneuver == "right":
            # Right-turn connectors are built from the curb (build_layout.py):
            # sublane k of the approach, the connector and the exit coincide.
            return min(lane, next_max)
        # Align all maneuvers to the median edge so discrete sublane IDs match
        # continuous lateral_shift across narrower connectors and exits.
        return min(next_max, max(0, lane + next_max - old_max))

    @staticmethod
    def _stop_position(vehicle):
        return (vehicle.approach_end - vehicle.length / 2 - STOP_MARGIN
                - vehicle.queue_stop_offset)

    def __init__(self, seed=1, flow_per_approach=2400.0,
                 motorbike_ratio=.85, p_left=.2, p_right=.2,
                 green_lvl=50.0, green_hdt=40.0, all_red=2.0,
                 roads_path=ROOT / "includes" / "roads.geojson",
                 demand_file=None, train_mode=False, mode="safe", yellow=4.0,
                 vehicle_params=None, startup_params=None,
                 junction_model="negotiation", negotiation_params=None):
        if mode not in ("safe", "gama_2025_6_4"):
            raise ValueError("mode must be 'safe' or 'gama_2025_6_4'")
        self.mode = mode
        self.seed = int(seed)
        numeric_parameters = (flow_per_approach, motorbike_ratio, p_left,
                              p_right, green_lvl, green_hdt, all_red)
        if (not all(math.isfinite(value) for value in numeric_parameters) or
                flow_per_approach < 0 or not 0 <= motorbike_ratio <= 1 or
                min(p_left, p_right, green_lvl, green_hdt, all_red) < 0 or
                p_left + p_right > 1 or flow_per_approach / 3600 * DT > 1):
            raise ValueError("Invalid flow, vehicle mix, turn split, or signal timing")
        self.rng = random.Random(seed)
        self.roads = road_lengths(roads_path)
        self._geometry_roads = {}
        source = json.loads(Path(roads_path).read_text(encoding="utf-8"))
        for feature in source["features"]:
            props = feature["properties"]
            self._geometry_roads[props["id"]] = GeometryRoad(
                feature["geometry"]["coordinates"], props["num_lanes"])
        base_conflicts = ConflictMap.from_geojson(source)
        self._lane_movement_defs = {}
        for movement in base_conflicts.movements.values():
            road = self._geometry_roads[movement.id]
            for occupied in (1, 2):
                for lane in range(max(0, road.num_lanes - occupied + 1)):
                    shift = (lane + (occupied - 1) / 2.0) * SUBLANE_WIDTH
                    count = max(1, math.ceil(road.length / 5.0))
                    points = tuple((road.pose(road.length * i / count, shift).x,
                                    road.pose(road.length * i / count, shift).y)
                                   for i in range(count + 1))
                    lengths = [0.0]
                    for a, b in zip(points, points[1:]):
                        lengths.append(lengths[-1] + math.dist(a, b))
                    lane_id = f"{movement.id}@{lane}:{occupied}"
                    self._lane_movement_defs[lane_id] = Movement(
                        lane_id, movement.arm, movement.to_arm, movement.turn,
                        points, tuple(lengths))
        self._conflict_map = base_conflicts
        self._lane_conflict_pair_cache = {}
        self._conflict_engine = ConflictEngine(
            self._conflict_map, left_wait_priority_after=10.0)
        self._junction_grants = set()
        self._road_index = None
        self._exited_this_tick = []
        # Upper bounds used to size index windows (updated on creation).
        self._max_vehicle_length = 4.5
        self._max_vehicle_gap = 1.0
        # Junction conflict model (safe mode only): "reservation" keeps the
        # time-window ConflictEngine; "negotiation" uses the SUMO-style
        # negotiation with forcing/group crossing in junction_negotiation.py.
        if junction_model not in ("reservation", "negotiation"):
            raise ValueError("junction_model must be 'reservation' or 'negotiation'")
        self.junction_model = junction_model
        self._junction_stop = {}
        self._negotiation = None
        if junction_model == "negotiation":
            self._negotiation = SimulatorAdapter(self, negotiation_params)
        self.flow_per_approach = flow_per_approach
        self.motorbike_ratio = motorbike_ratio
        self.p_left = p_left
        self.p_right = p_right
        self.green_lvl = green_lvl
        self.green_hdt = green_hdt
        self.all_red = all_red
        # Yellow interval (s), carved from the end of each green so the cycle
        # length and green onsets are unchanged; defaults to the fixed 4 s plan.
        yellow = float(yellow)
        if (not math.isfinite(yellow) or yellow < 0 or
                yellow > min(green_lvl, green_hdt) + 1e-9):
            raise ValueError("yellow must be within [0, min(green_lvl, green_hdt)]")
        self._yellow = yellow
        self.yellow_phase = None
        self._yellow_from = None
        self._yellow_epoch = 0
        self._last_yellow = None
        self.yellow_ticks_left = 0
        self.entries_during_yellow = 0
        self.entries_during_red = 0
        self.permitted_right_on_red_entries = 0
        self.cycle = 0
        self.phase = "LVL"
        self.vehicles = []
        self._next_creation_id = 0
        self.vehicle_params = {kind: dict(values)
                               for kind, values in DEFAULT_VEHICLE_PARAMS.items()}
        for kind, values in (vehicle_params or {}).items():
            if kind not in self.vehicle_params:
                raise ValueError(f"unknown vehicle type {kind!r}")
            unknown = set(values) - set(self.vehicle_params[kind])
            if unknown:
                raise ValueError(f"unknown vehicle parameters {sorted(unknown)}")
            if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values.values()):
                raise ValueError("vehicle parameters must be positive")
            self.vehicle_params[kind].update({k: float(v) for k, v in values.items()})
        self.startup_params = dict(startup_params or {})
        if self.mode == "safe":
            self._startup_schedulers = {
                arm: SafeStartupScheduler(seed=self.seed, **self.startup_params)
                for arm in ARMS
            }
            # The simulation starts in LVL, so NE/SW have no red-to-green
            # startup event at cycle 1. Mark those schedulers green now; after
            # their first red phase, normal onset snapshots apply.
            for arm in ("NE", "SW"):
                self._startup_schedulers[arm].update(0.0, True, ())
        self.created = self.passed = 0
        self.front_row_max = {arm: 0 for arm in ARMS}
        self.queue_time_integral = 0.0
        self.system_time_integral = 0.0
        self.weighted_delay_integral = 0.0
        self.road_safety_corrections = 0
        self.max_road_safety_correction = 0.0
        self.largest_road_safety_event = None
        self.green_ticks = 0
        self.train_mode = bool(train_mode)
        self.dqn_waiting_action = False
        self.dqn_action = -1
        self.next_decision_in_cycles = 50
        self.red_ticks_left = 0
        self.pending_phase = "LVL"
        self._phase_change_pending = False
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
                        if (row[0] < 1 or not 0 <= row[1] < 4 or
                                row[2] not in (0, 1) or
                                not math.isfinite(row[3]) or row[3] <= 0 or
                                not 0 <= row[4] <= 2):
                            raise ValueError(f"Invalid demand row: {raw}")
                        occupied = 2 if row[2] else 1
                        incoming = self.roads[f"{ARMS[row[1]]}_in"]
                        if not 0 <= row[5] <= incoming["num_lanes"] - occupied:
                            raise ValueError(f"Invalid demand lane: {raw}")
                        self._rows.append(row)
            keys = [(row[0], row[1]) for row in self._rows]
            if any(left > right for left, right in zip(keys, keys[1:])):
                raise ValueError("Demand rows must be ordered by cycle and arm")

    def _phase(self, cycle=None):
        return self._signal(cycle)[0]

    def _signal(self, cycle=None):
        """(phase, yellow_of) of the fixed plan.

        ``phase`` is "LVL"/"HDT" while green and "none" otherwise (yellow and
        all-red), so code reading ``phase`` keeps its old meaning;
        ``yellow_of`` names the phase whose yellow is showing.
        """
        # GAMA's ``time`` at cycle n is (n - 1) * step inside the reflex.
        cycle = self.cycle if cycle is None else cycle
        t = max(0, cycle - 1) * DT
        length = self.green_lvl + self.green_hdt + 2 * self.all_red
        if length <= 0:
            return "none", None
        t %= length
        yellow = getattr(self, "_yellow", 0.0)
        if t < self.green_lvl - yellow:
            return "LVL", None
        if t < self.green_lvl:
            return "none", "LVL"
        if t < self.green_lvl + self.all_red:
            return "none", None
        if t < self.green_lvl + self.all_red + self.green_hdt - yellow:
            return "HDT", None
        if t < self.green_lvl + self.all_red + self.green_hdt:
            return "none", "HDT"
        return "none", None

    def _update_last_yellow(self):
        if self.yellow_phase is not None:
            self._last_yellow = self.yellow_phase
        elif self.phase in ("LVL", "HDT"):
            self._last_yellow = None

    def set_action(self, action):
        if not self.train_mode or not self.dqn_waiting_action or action not in (0, 2):
            raise ValueError("Action requires a Le Van Luong decision tick and phase 0 or 2")
        self.dqn_action = action
        self.dqn_waiting_action = False

    def _advance_signal(self):
        previous_yellow = self.yellow_phase
        if not self.train_mode:
            self.phase, self.yellow_phase = self._signal()
            if self.yellow_phase is not None and previous_yellow is None:
                self._yellow_epoch += 1
            self._update_last_yellow()
            if self.phase in ("LVL", "HDT"):
                self.green_ticks += 1
            else:
                self.green_ticks = 0
            return
        if self.dqn_action >= 0:
            wanted = "LVL" if self.dqn_action == 0 else "HDT"
            if (wanted != self.phase and self.red_ticks_left == 0 and
                    self.yellow_ticks_left == 0):
                self.pending_phase = wanted
                self._phase_change_pending = True
                self._yellow_from = self.phase if self.phase in ("LVL", "HDT") else None
                self.yellow_ticks_left = (int(round(self._yellow / DT))
                                          if self._yellow_from else 0)
                self.red_ticks_left = int(round(self.all_red / DT))
            self.dqn_action = -1
        if self.yellow_ticks_left > 0:
            # Yellow of the phase being left, then the all-red clearance.
            self.phase = "none"
            self.yellow_phase = self._yellow_from
            if previous_yellow is None:
                self._yellow_epoch += 1
            self.yellow_ticks_left -= 1
        elif self.red_ticks_left > 0:
            self.yellow_phase = None
            self.phase = "none"
            self.red_ticks_left -= 1
        elif self._phase_change_pending:
            self.phase = self.pending_phase
            self.yellow_phase = None
            self._phase_change_pending = False
        self._update_last_yellow()
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
            minimum, maximum = (0, incoming["num_lanes"] - occupied) if turn == "through" else turn_lane_bounds(incoming["num_lanes"], occupied, turn)
            lane = self.rng.randrange(minimum, maximum + 1)
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
        if (not 0 <= lane <= incoming["num_lanes"] - occupied or
                not math.isfinite(speed_kmh) or speed_kmh <= 0):
            raise ValueError("Invalid lane or requested speed")
        # Through traffic can enter from any sublane; it must merge before
        # joining the signal queue. Turning requests keep their turn corridor.
        if turn != "through":
            lane_min, lane_max = turn_lane_bounds(incoming["num_lanes"], occupied, turn)
            lane = min(max(lane, lane_min), lane_max)
        approach_end = incoming["length"]
        connector_end = approach_end + connector["length"]
        initial_shift = (lane + (occupied - 1) / 2.0) * SUBLANE_WIDTH
        length = 4.5 if is_car else 1.9
        kind_params = self.vehicle_params["car" if is_car else "motorbike"]
        position = 0.0
        new_half_width = 0.85 if is_car else 0.35
        for other in self.vehicles if self.mode == "safe" else ():
            if other.road != f"{arm}_in":
                continue
            if abs(initial_shift - other.current_lateral_shift) >= (
                    new_half_width + self._half_width(other) + LATERAL_CLEARANCE):
                continue
            position = min(position, other.position - (length + other.length) / 2
                           - max(1.0, other.gap))
        creation_id = self._next_creation_id
        self._next_creation_id += 1
        # Fixed for each vehicle and seed; do not consume the demand RNG.
        queue_rng = random.Random((self.seed * 1000003) ^ (creation_id * 9176) ^ ARMS.index(arm))
        queue_stop_offset = (queue_rng.uniform(0.0, 1.8 if not is_car else 1.2)
                             if self.mode == "safe" else 0.0)
        route_min, route_max = turn_lane_bounds(incoming["num_lanes"], occupied, turn)
        initial_merge_ready = (self.mode == "safe" and turn == "through" and
                               not route_min <= lane <= route_max)
        self._max_vehicle_length = max(self._max_vehicle_length, length)
        self._max_vehicle_gap = max(self._max_vehicle_gap, kind_params["gap"])
        self.vehicles.append(Vehicle(
            arm=arm, to_arm=target, maneuver=turn, lane=lane, occupied=occupied,
            length=length, desired=speed_kmh / 3.6,
            accel=kind_params["accel"], decel=kind_params["decel"],
            gap=kind_params["gap"], headway=kind_params["headway"],
            approach_end=approach_end,
            connector_end=connector_end,
            route_end=connector_end + outgoing["length"],
            road=("" if self.mode == "gama_2025_6_4" else f"{arm}_in"),
            creation_id=creation_id,
            created_cycle=self.cycle,
            position=position,
            demand_row_id=demand_row_id,
            current_lateral_shift=initial_shift,
            start_lateral_shift=initial_shift,
            target_lateral_shift=initial_shift,
            lane_change_progress=1.0,
            lateral_duration=0.6 if not is_car else 1.2,
            queue_stop_offset=queue_stop_offset,
            time_since_lane_change=0.8 if initial_merge_ready else 0.0,
            gama_entered=self.mode != "gama_2025_6_4"))
        self.created += 1

    def _destination_has_room(self, vehicle):
        """Require the complete vehicle plus a buffer to fit beyond the connector."""
        target = vehicle.to_arm
        target_road = f"{target}_out"
        out_length = self.roads[target_road]["length"]
        target_shift = (self._mapped_lane(vehicle, vehicle.lane,
                                          f"{vehicle.arm}_in",
                                          f"{vehicle.arm}_{vehicle.to_arm}") +
                        (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
        groups = getattr(self, "_dest_groups", None)
        pool = (groups.get(target, ()) if groups is not None else self.vehicles)
        for other in pool:
            if other is vehicle or other.to_arm != target:
                continue
            other_road, _ = self._road_state(other)
            # Express the other vehicle's center relative to this movement's
            # connector/exit boundary. Vehicles still on a connector may
            # already occupy the first metres of the common exit.
            if other_road == target_road:
                other_progress = other.position - other.connector_end
            elif other_road == f"{other.arm}_{target}":
                other_progress = other.position - other.connector_end
            else:
                continue
            other_shift = other.current_lateral_shift
            if abs(target_shift - other_shift) >= (
                    self._half_width(vehicle) + self._half_width(other) + LATERAL_CLEARANCE):
                continue
            # A vehicle still approaching the exit does not occupy downstream
            # space until its front reaches the connector/exit seam.
            if other_progress + other.length / 2 <= 0.0:
                continue
            if (other_progress - other.length / 2 <
                    vehicle.length / 2 + max(1.5, vehicle.gap) + max(0.0, vehicle.speed * 0.4)):
                return False
        # Also reject routes too short to contain the whole body at the entry.
        return out_length >= vehicle.length + max(1.0, vehicle.gap)

    def _lane_movement_id(self, vehicle, on_connector=False):
        base = f"{vehicle.arm}_{vehicle.to_arm}"
        lane = (vehicle.lane if on_connector else self._mapped_lane(
            vehicle, vehicle.lane, f"{vehicle.arm}_in", base))
        return f"{base}@{lane}:{vehicle.occupied}"

    @staticmethod
    def _leader_advance(leader, road, moved_ids):
        """Distance the leader is still expected to cover this tick.

        A leader processed earlier in this sequential pass has already moved
        and its lane-map entry holds the new position, so adding its advance
        again would let the follower encroach and be pushed back afterwards.
        """
        if (leader is None or id(leader) in moved_ids or leader.road != road or
                leader.speed <= 0.5):
            return 0.0
        return max(0.0, leader.speed * DT - 0.5 * leader.decel * DT * DT)

    def _new_road_lanes(self, road):
        if self.mode == "gama_2025_6_4":
            # GAMA parity mode never uses the safe-mode neighbour scans.
            return [[] for _ in range(self.roads[road]["num_lanes"])]
        index = self._road_index.setdefault(road, _RoadIndex())
        return [_Sublane(index) for _ in range(self.roads[road]["num_lanes"])]

    def _nearby(self, road, low, high):
        """Superset of vehicles that may lie on ``road`` within [low, high]."""
        index = self._road_index.get(road)
        found = (index.window(low - INDEX_MARGIN, high + INDEX_MARGIN)
                 if index is not None else [])
        return found + self._exited_this_tick if self._exited_this_tick else found

    @staticmethod
    def _moving_away(current, target, other_lo, other_hi):
        """True when a lateral move from ``current`` to ``target`` heads away
        from a body occupying ``[other_lo, other_hi]`` (never closer to it).

        Lateral clearance checks must only block moves that reduce the gap;
        otherwise two bodies that are already close block each other's
        escape and freeze half-way lane changes indefinitely.
        """
        if target > current:
            return other_hi <= current
        if target < current:
            return other_lo >= current
        return False

    def _can_enter_lane(self, vehicle):
        """Whether the vehicle's current sublane lets it cross the stop line.

        Single source for the lane condition used by ``can_cross`` in
        ``_move`` and for issuing junction requests/entry tokens.
        """
        return (self.mode == "gama_2025_6_4" or
                self._route_target_lane(vehicle) == vehicle.lane)

    @staticmethod
    def _arm_in_phase(arm, phase):
        return ((arm in ("NE", "SW") and phase == "LVL") or
                (arm in ("SE", "NW") and phase == "HDT"))

    def _arm_is_yellow(self, arm):
        return ((arm in ("NE", "SW") and self.yellow_phase == "LVL") or
                (arm in ("SE", "NW") and self.yellow_phase == "HDT"))

    def _green_for(self, vehicle):
        """Whether this vehicle may proceed through its stop line now.

        Green for its arm, or yellow with a "go" decision. In GAMA mode (no
        yellow) this is exactly ``_arm_is_green``.
        """
        # The site's right-turn movement is permissive on red. It still needs
        # a junction grant and downstream room; signal permission alone never
        # authorizes entry into a conflicting through stream.
        if self.mode == "safe" and vehicle.maneuver == "right":
            return True
        if self._arm_is_green(vehicle.arm):
            return True
        if self.yellow_phase is None or not self._arm_is_yellow(vehicle.arm):
            # A "go" decision taken during the last yellow stays valid through
            # the following all-red (the vehicle runs the early red, as
            # observed by Chauhan et al.); it lapses at the next green onset.
            return (self.phase == "none" and self._last_yellow is not None and
                    self._arm_in_phase(vehicle.arm, self._last_yellow) and
                    vehicle.yellow_epoch == self._yellow_epoch and vehicle.yellow_go)
        if vehicle.yellow_epoch != self._yellow_epoch:
            vehicle.yellow_go = self._decide_stop_or_go(vehicle)
            vehicle.yellow_epoch = self._yellow_epoch
        return vehicle.yellow_go

    def stop_or_go_probability(self, vehicle):
        """P(pass) of the Chauhan et al. (2022) point-data logit."""
        c = CHAUHAN2022_PDM
        x = max(0.0, vehicle.approach_end - vehicle.position - vehicle.length / 2.0)
        utility = c["constant"] + c["speed"] * max(0.0, vehicle.speed) + c["distance"] * x
        if vehicle.occupied == 2:
            utility += c["car"]
        width = self.roads[vehicle.in_road]["num_lanes"] * SUBLANE_WIDTH
        fraction = vehicle.current_lateral_shift / max(width, 1e-9)
        # Sublanes are counted from the right (kerb) side: thirds of the
        # carriageway map to kerb / intermediate / median-side lanes.
        if fraction >= 2.0 / 3.0:
            utility += c["median_lane"]
        elif fraction >= 1.0 / 3.0:
            utility += c["intermediate_lane"]
        return 1.0 / (1.0 + math.exp(-utility))

    def _decide_stop_or_go(self, vehicle):
        if vehicle.road != vehicle.in_road:
            return True
        stop = self._stop_position(vehicle)
        if vehicle.position > stop + 1e-8:
            return True
        x = vehicle.approach_end - vehicle.position - vehicle.length / 2.0
        if x > STOP_GO_MAX_DISTANCE:
            return False
        # A permissive yellow decision is only meaningful if the vehicle can
        # reach the junction before the opposing green. Otherwise a distant
        # vehicle can keep full speed throughout yellow and all-red, then be
        # stopped in one tick when its grant expires.
        clearance_time = self._yellow + self.all_red
        speed_limit = min(vehicle.desired, 60.0 / 3.6)
        cruise_speed = max(0.0, min(vehicle.speed, speed_limit))
        accel_time = min(clearance_time,
                         max(0.0, speed_limit - cruise_speed) /
                         max(0.1, vehicle.accel))
        reachable = (cruise_speed * accel_time +
                     0.5 * vehicle.accel * accel_time * accel_time +
                     speed_limit * (clearance_time - accel_time))
        if x > reachable - 1.0:
            return False
        room = max(0.05, stop - vehicle.position)
        if vehicle.speed * vehicle.speed / (2.0 * room) > STOP_GO_MAX_DECEL:
            return True
        # Deterministic per vehicle and yellow interval; never consumes the
        # demand RNG.
        draw = random.Random((self.seed * 7919) ^ (vehicle.creation_id * 104729)
                             ^ (self._yellow_epoch * 1299709)).random()
        return draw < self.stop_or_go_probability(vehicle)

    def _arm_is_green(self, arm):
        return ((arm in ("NE", "SW") and self.phase == "LVL") or
                (arm in ("SE", "NW") and self.phase == "HDT"))

    def _reserve_junction(self):
        """Atomically arbitrate all green approach requests and refresh occupants."""
        now = self.cycle * DT
        live = {str(v.creation_id): v for v in self.vehicles}
        # Vehicles grouped by destination arm for ``_destination_has_room``;
        # nothing moves inside this method, so the grouping stays exact.
        self._dest_groups = {}
        for item in self.vehicles:
            self._dest_groups.setdefault(item.to_arm, []).append(item)
        # An entry token is a grant to cross the stop line while the arm is
        # green. If the signal leaves green before the front crosses the line
        # (all-red/clearance or a hold at the end of green) the grant is void.
        # Keeping it would leave a motionless "committed" phantom on the stop
        # line that blocks every conflicting movement of the next phase.
        # The grant is also void while the vehicle is outside its route
        # corridor (it cannot legally cross, see ``_can_enter_lane``) or has
        # stood still for TOKEN_STALL_LIMIT s: a motionless committed request
        # would otherwise block every conflicting movement indefinitely.
        for vehicle in self.vehicles:
            if not (vehicle.junction_entry_token and vehicle.road == f"{vehicle.arm}_in"
                    and vehicle.position <= self._stop_position(vehicle) + 1e-8):
                vehicle.token_stall_time = 0.0
                continue
            vehicle.token_stall_time = (vehicle.token_stall_time + DT
                                        if vehicle.speed < 0.1 else 0.0)
            stalled = vehicle.token_stall_time >= TOKEN_STALL_LIMIT - 1e-9
            if (not self._green_for(vehicle) or
                    not self._can_enter_lane(vehicle) or stalled):
                if stalled:
                    # Step aside for one second so conflicting traffic can
                    # use the window this vehicle failed to use.
                    vehicle.token_cooldown_until = now + TOKEN_STALL_COOLDOWN
                vehicle.junction_entry_token = False
                vehicle.junction_wait_since = None
                vehicle.token_stall_time = 0.0
        for vehicle_id, reservation in tuple(self._conflict_engine.reservations.items()):
            vehicle = live.get(vehicle_id)
            if (vehicle is None or
                    vehicle.position - vehicle.connector_end >= vehicle.length / 2):
                self._conflict_engine.release(vehicle_id)
        prev_grants = set(self._junction_grants)
        requests = []
        for vehicle in self.vehicles:
            road, _ = self._road_state(vehicle)
            base_movement_id = f"{vehicle.arm}_{vehicle.to_arm}"
            if road == base_movement_id:
                movement_id = self._lane_movement_id(vehicle, on_connector=True)
                connector_s = max(0.0, vehicle.position - vehicle.approach_end)
                requests.append(MovementRequest(
                    vehicle_id=str(vehicle.creation_id), movement_id=movement_id,
                    speed=max(0.1, vehicle.speed), vehicle_length=vehicle.length,
                    turn=vehicle.maneuver, committed=True,
                    connector_s=connector_s, downstream_clear=True,
                    waiting_since=vehicle.junction_wait_since,
                    accel=vehicle.accel, desired_speed=vehicle.desired,
                    speed0=max(0.0, vehicle.speed), entry_distance=0.0))
                continue
            if road != f"{vehicle.arm}_in":
                continue
            movement_id = self._lane_movement_id(vehicle)
            green = self._green_for(vehicle)
            stop_position = self._stop_position(vehicle)
            committed_approach = vehicle.position > stop_position + 1e-8
            physically_entered = (vehicle.position + vehicle.length / 2 >=
                                  vehicle.approach_end)
            if not (committed_approach or vehicle.junction_entry_token) and not green:
                vehicle.junction_wait_since = None
                continue
            if not committed_approach and (not self._can_enter_lane(vehicle) or
                                           now < vehicle.token_cooldown_until):
                # Still merging into its route corridor, or yielding after a
                # stalled grant: no junction request this tick.
                continue
            distance = max(0.0, (vehicle.approach_end if
                                 (committed_approach or vehicle.junction_entry_token)
                                 else stop_position) - vehicle.position)
            req_horizon = (max(28.0, vehicle.speed * vehicle.speed / (2.0 * max(1.0, vehicle.decel))
                               + vehicle.speed * 0.8 + 4.0)
                           if vehicle.speed > 3.0 else max(8.0, vehicle.speed * 2.0 + 2.0))
            if (not committed_approach and not vehicle.junction_entry_token and
                    distance > req_horizon):
                continue
            approach_speed = max(0.0, vehicle.speed)
            # Predict entry ETA using realistic startup acceleration from rest so
            # stopped front-of-line vehicles are not endlessly overtaken by faster
            # distant vehicles, and do not inflate connector occupancy at 2.0 m/s.
            entry_offset = (0.0 if (committed_approach or vehicle.junction_entry_token) else
                            max(0.0, vehicle.approach_end - stop_position))
            rem = distance + entry_offset
            a_eff = max(0.5, vehicle.accel)
            t_accel = (-approach_speed + math.sqrt(
                approach_speed * approach_speed + 2.0 * a_eff * rem)) / a_eff
            arrival = now + (min(t_accel, rem / approach_speed)
                             if approach_speed > 0.5 else t_accel)
            braking_dist = approach_speed * approach_speed / (2.0 * max(1.0, vehicle.decel))
            dilemma_hold = (not committed_approach and not vehicle.junction_entry_token and
                            vehicle.creation_id in prev_grants and approach_speed >= 4.0 and
                            distance <= braking_dist and
                            vehicle.leading_distance > distance + 2.0)
            requests.append(MovementRequest(
                vehicle_id=str(vehicle.creation_id), movement_id=movement_id,
                approach_distance=distance, speed=max(4.5, approach_speed * 0.9),
                vehicle_length=vehicle.length, turn=vehicle.maneuver,
                committed=((committed_approach or vehicle.junction_entry_token or dilemma_hold)
                           and (vehicle.maneuver != "right" or physically_entered)),
                downstream_clear=(True if (dilemma_hold or
                                           ((committed_approach or vehicle.junction_entry_token)
                                            and vehicle.maneuver != "right")) else
                                  self._destination_has_room(vehicle)),
                arrival_time=arrival, waiting_since=vehicle.junction_wait_since,
                accel=vehicle.accel, desired_speed=vehicle.desired,
                speed0=max(0.0, vehicle.speed), entry_distance=rem))
        lane_map = {r.movement_id: self._lane_movement_defs[r.movement_id]
                    for r in requests if r.movement_id in self._lane_movement_defs}
        lane_conflicts = object.__new__(ConflictMap)
        lane_conflicts.movements = lane_map
        lane_conflicts.clearance = 1.0
        lane_conflicts.default_vehicle_length = 4.5
        lane_conflicts.near_miss_distance = 1.85
        lane_conflicts.conflicts = {}
        lane_conflicts.near_miss_pairs = set()
        ids = sorted(lane_map)
        for i, aid in enumerate(ids):
            for bid in ids[i + 1:]:
                key = (aid, bid)
                if key not in self._lane_conflict_pair_cache:
                    base_a, spec_a = aid.split("@", 1)
                    lane_a, occ_a = (int(x) for x in spec_a.split(":"))
                    base_b, spec_b = bid.split("@", 1)
                    lane_b, occ_b = (int(x) for x in spec_b.split(":"))
                    if base_a == base_b:
                        if lane_a < lane_b + occ_b and lane_b < lane_a + occ_a:
                            m = lane_map[aid]
                            self._lane_conflict_pair_cache[key] = ConflictMap._pair(
                                lane_conflicts, m, lane_map[bid])[0]
                        else:
                            self._lane_conflict_pair_cache[key] = ()
                    else:
                        half_a = 0.85 if occ_a == 2 else 0.35
                        half_b = 0.85 if occ_b == 2 else 0.35
                        lane_conflicts.near_miss_distance = half_a + half_b + 0.15
                        self._lane_conflict_pair_cache[key] = ConflictMap._pair(
                            lane_conflicts, lane_map[aid], lane_map[bid])[0]
                if self._lane_conflict_pair_cache[key]:
                    lane_conflicts.conflicts[key] = self._lane_conflict_pair_cache[key]
        # Conflict adjacency so the engine visits only movements that
        # actually conflict (identical windows, far fewer interval lookups).
        adjacent = {}
        for aid, bid in lane_conflicts.conflicts:
            adjacent.setdefault(aid, []).append(bid)
            adjacent.setdefault(bid, []).append(aid)
        lane_conflicts.adjacent = adjacent
        self._conflict_engine.map = lane_conflicts
        decisions = self._conflict_engine.decide_batch(requests, now=now)
        self._junction_grants = {
            int(vehicle_id) for vehicle_id, decision in decisions.items()
            if decision.allowed
        }
        # Entry ETAs are predictions. Keep a fresh approach from entering while
        # a committed body physically occupies one of its conflict intervals.
        # Once its rear leaves that interval, compatible traffic may proceed.
        active = [r for r in requests if r.connector_s is not None]
        right_committed = [r for r in requests if r.connector_s is None and r.committed and r.turn == "right"]
        for request in requests:
            if request.connector_s is not None or request.committed:
                continue
            blocked = False
            pool = active + right_committed if request.turn == "right" else active
            for occupant in pool:
                occ_s = occupant.connector_s if occupant.connector_s is not None else 0.0
                if occupant.movement_id == request.movement_id:
                    # Same lane on the same connector forms a platoon; only hold
                    # while the leader is still occupying the entry zone.
                    if occ_s < occupant.vehicle_length + 2.5:
                        blocked = True
                        break
                elif occupant.movement_id.split("@", 1)[0] == request.movement_id.split("@", 1)[0]:
                    # Adjacent sublanes entering the same connector (especially
                    # narrower right turns) must clear the seam funnel safely.
                    # Tight-radius right turns also compress world arc distance
                    # on inner sublanes, so widen and extend lateral stagger.
                    clear_s = 25.0 if request.turn == "right" else occupant.vehicle_length + 2.5
                    if occ_s < clear_s:
                        req_v = live.get(request.vehicle_id)
                        occ_v = live.get(occupant.vehicle_id)
                        if req_v is not None and occ_v is not None and (
                                occupant.connector_s is not None or occ_v.position > req_v.position):
                            req_shift = self._continuous_seam_pose(
                                f"{req_v.arm}_in", f"{req_v.arm}_{req_v.to_arm}",
                                req_v.current_lateral_shift)[1]
                            occ_lo = min(occ_v.current_lateral_shift, occ_v.target_lateral_shift)
                            occ_hi = max(occ_v.current_lateral_shift, occ_v.target_lateral_shift)
                            lat_gap = max(occ_lo - req_shift, req_shift - occ_hi, 0.0)
                            margin = 1.2 if request.turn == "right" else 0.2
                            if lat_gap < (self._half_width(req_v) + self._half_width(occ_v) + margin):
                                blocked = True
                                break
                elif occupant.connector_s is not None:
                    if occupant.connector_s < occupant.vehicle_length + 4.5:
                        req_v = live.get(request.vehicle_id)
                        occ_v = live.get(occupant.vehicle_id)
                        if req_v is not None and occ_v is not None and req_v.arm == occ_v.arm:
                            req_pose = self._geometry_roads[f"{req_v.arm}_in"].pose(
                                req_v.approach_end, req_v.current_lateral_shift)
                            occ_pose = self._geometry_roads[f"{occ_v.arm}_{occ_v.to_arm}"].pose(
                                occupant.connector_s, occ_v.current_lateral_shift)
                            dx, dy = occ_pose.x - req_pose.x, occ_pose.y - req_pose.y
                            along = dx * req_pose.tx + dy * req_pose.ty
                            lat = abs(dx * req_pose.ty - dy * req_pose.tx)
                            if (along > -occ_v.length / 2.0 and along < req_v.length / 2.0 + occ_v.length / 2.0 + 3.0
                                    and lat < self._half_width(req_v) + self._half_width(occ_v) + 0.7):
                                blocked = True
                                break
            if blocked:
                self._junction_grants.discard(int(request.vehicle_id))
        denied_ids = (set(decisions) - {str(i) for i in self._junction_grants})
        for vehicle_id in denied_ids:
            vehicle = live.get(vehicle_id)
            if vehicle is not None and vehicle.road.endswith("_in"):
                if vehicle.junction_wait_since is None:
                    vehicle.junction_wait_since = now
        for vehicle_id in self._junction_grants:
            vehicle = live.get(str(vehicle_id))
            if vehicle is not None:
                vehicle.junction_wait_since = None
                if (vehicle.road.endswith("_in") and self._can_enter_lane(vehicle) and
                        vehicle.position + max(0.0, vehicle.speed) * DT >=
                        self._stop_position(vehicle)):
                    vehicle.junction_entry_token = True
        self._dest_groups = None

    def _move(self):
        starting_positions = {v.creation_id: v.position for v in self.vehicles}
        starting_speeds = {v.creation_id: v.speed for v in self.vehicles}
        # GAMA iterates agents in insertion order. Each agent performs its
        # lane choice and movement before the next agent reads its neighbors.
        lane_maps = {}
        self._road_index = {}
        self._exited_this_tick = []
        self._free_idm_cache = {}
        self._neighbor_road_state = {}
        ticks_to_green_end = None
        startup_allowed = {}
        startup_gate_enabled = {}
        if self.mode == "safe":
            from_startup = self._startup_schedulers
            for arm in ARMS:
                green_for_arm = (
                    (arm in ("NE", "SW") and self.phase == "LVL") or
                    (arm in ("SE", "NW") and self.phase == "HDT")
                )
                startup_gate_enabled[arm] = green_for_arm
                snapshot = []
                if green_for_arm:
                    for item in self.vehicles:
                        if (item.arm == arm and item.road == f"{arm}_in" and
                                item.created_cycle < self.cycle):
                            stop = self._stop_position(item)
                            snapshot.append(StartupVehicle(
                                vehicle_id=item.creation_id,
                                road_id=arm,
                                sublane=item.lane,
                                position=item.position,
                                speed=item.speed,
                                stop_line_position=stop,
                                committed=item.position > stop + 1e-8,
                                occupied=item.occupied,
                            ))
                startup_allowed[arm] = from_startup[arm].update(
                    self.cycle * DT, green_for_arm, snapshot)
        if self.phase in ("LVL", "HDT") and self._yellow <= 0.0:
            # Without a yellow interval vehicles are given perfect foresight of
            # the end of green (legacy behaviour). With a yellow they react
            # to the yellow onset instead (stop-or-go decision).
            if self.train_mode:
                # The policy can change phase immediately after every 50th
                # tick. Keep the stop-line buffer clear at that decision point
                # even when the next action has not been chosen yet.
                ticks_to_green_end = (50 - self.cycle % 50) % 50 + 1
            else:
                # The fixed plan is periodic: remember the next phase change
                # instead of rescanning up to 1000 future ticks every tick.
                cached = getattr(self, "_next_phase_change", None)
                if (cached is not None and cached[1] == self.phase and
                        cached[2] == (self.green_lvl, self.green_hdt, self.all_red) and
                        cached[3] <= self.cycle < cached[0]):
                    ticks_to_green_end = cached[0] - self.cycle
                else:
                    for offset in range(1, 1001):
                        if self._phase(self.cycle + offset) != self.phase:
                            ticks_to_green_end = offset
                            self._next_phase_change = (
                                self.cycle + offset, self.phase,
                                (self.green_lvl, self.green_hdt, self.all_red),
                                self.cycle)
                            break
        if self.mode == "safe":
            if self._negotiation is not None:
                self._negotiation.step(DT)
            else:
                self._reserve_junction()
        for vehicle in self.vehicles:
            pending_entry = self.mode == "gama_2025_6_4" and not vehicle.gama_entered
            if pending_entry:
                road, progress = f"{vehicle.arm}_in", 0.0
            else:
                road, progress = self._road_state(vehicle)
            self._neighbor_road_state[id(vehicle)] = (road, progress)
            if road not in lane_maps:
                lane_maps[road] = self._new_road_lanes(road)
            if pending_entry:
                continue
            max_lane = max(0, self.roads[road]["num_lanes"] - vehicle.occupied)
            if vehicle.lane > max_lane:
                vehicle.lane = max_lane
            entry = (progress, id(vehicle), vehicle)
            for sub in self._footprint(vehicle):
                bisect.insort(lane_maps[road][sub], entry)
        survivors = []
        moved_ids = set()
        for vehicle in self.vehicles:
            moved_ids.add(id(vehicle))
            old_road, old_progress = self._neighbor_road_state[id(vehicle)]
            is_in_road = old_road.endswith("_in")
            green = self._green_for(vehicle) if is_in_road else True

            # GAMA DrivingSkill.moveAcrossSegments:
            # If distanceToNextPoint < 0.01 at the start of the step, the agent
            # snaps directly to the node (rem = 0.0). If red (!readyToCross),
            # speed is set to 0.0 and movement terminates without lane change.
            stop_position = (vehicle.approach_end if self.mode == "gama_2025_6_4"
                             else self._stop_position(vehicle)) if is_in_road else None
            committed = (is_in_road and vehicle.position > stop_position + 1e-8)
            route_invalid = (self.mode == "safe" and is_in_road and not committed
                             and vehicle.lane != self._route_target_lane(vehicle))
            if not route_invalid:
                vehicle.merge_wait = 0.0
                vehicle.merge_claim_shift = math.nan
            elif vehicle.speed < 1.0:
                vehicle.merge_wait += DT
            if route_invalid:
                # Reserve distance for each remaining sublane change. A through
                # vehicle may enter at the curb, but cannot reach the signal
                # queue until its route corridor is clear.
                remaining_lanes = abs(vehicle.lane - self._route_target_lane(vehicle))
                stop_position -= 12.0 * (remaining_lanes + 1)
            safe_committed = (committed and
                              (self.mode != "safe" or vehicle.junction_entry_token))
            entry_horizon = (max(28.0, vehicle.speed * vehicle.speed / (2.0 * max(1.0, vehicle.decel))
                                 + vehicle.speed * 0.8 + 4.0)
                             if vehicle.speed > 3.0 else max(8.0, vehicle.speed * 2.0 + 2.0))
            denied_entry = (self.mode == "safe" and is_in_road and green and
                            not committed and not vehicle.junction_entry_token and
                            vehicle.creation_id not in self._junction_grants and
                            self._stop_position(vehicle) - vehicle.position <= entry_horizon)
            if (self.mode == "safe" and is_in_road and not committed and
                    startup_gate_enabled.get(vehicle.arm, False) and
                    vehicle.creation_id not in startup_allowed.get(vehicle.arm, ())):
                # Startup gating applies only to the onset queue recorded by
                # the arm scheduler. Keep the vehicle at its existing position
                # before integrating acceleration or distance.
                vehicle.speed = 0.0
                self._free_idm_cache.pop(id(vehicle), None)
                survivors.append(vehicle)
                continue
            if (is_in_road and not green and not committed and not route_invalid
                    and stop_position - vehicle.position < 0.01):
                # The vehicle is at the stop point. Keep its lane-map entry at
                # the actual position so later vehicles see the same gap.
                vehicle.speed = 0.0
                self._free_idm_cache.pop(id(vehicle), None)
                survivors.append(vehicle)
                continue

            lane_decision = self._choose_lane(vehicle, lane_maps)
            move_lane = vehicle.lane
            leader, gap, _, _ = self._neighbors(vehicle, vehicle.lane, lane_maps)
            if self.mode == "safe":
                lane_centre = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                if abs(vehicle.current_lateral_shift - lane_centre) > 1e-6:
                    # Same lateral predicate as _resolve_road_conflicts: the
                    # body ahead of the vehicle's actual lateral position.
                    body_leader, body_gap, _, _ = self._neighbors(
                        vehicle, vehicle.lane, lane_maps,
                        shift=vehicle.current_lateral_shift)
                    leader, gap = body_leader, body_gap
            if safe_committed and not green and (leader is None or leader.road != old_road):
                # A vehicle whose front crossed the stop line on green must
                # clear the approach during all-red. Connector conflicts are
                # handled separately; they must not trap it on the stop line.
                leader, gap = None, math.inf
            if (self.mode == "safe" and
                    (is_in_road or old_road == vehicle.conn_road) and
                    self.junction_model == "negotiation"):
                # Keep following a same-approach body across the stop-line
                # seam, even when it takes another connector. Those paths
                # share the first metres of the junction and only diverge
                # after their bodies have cleared one another.
                for ahead in self.vehicles:
                    if (ahead is vehicle or ahead.arm != vehicle.arm or
                            ahead.position <= vehicle.position or
                            ahead.position - vehicle.position > 40.0 or
                            ahead.road not in (vehicle.in_road, ahead.conn_road) or
                            abs(ahead.current_lateral_shift -
                                vehicle.current_lateral_shift) >=
                            self._half_width(vehicle) + self._half_width(ahead) +
                            LATERAL_CLEARANCE):
                        continue
                    if ahead.conn_road != vehicle.conn_road:
                        own_band = self._negotiation._band_path(
                            vehicle, vehicle.conn_road,
                            old_road == vehicle.conn_road)
                        lead_band = self._negotiation._band_path(
                            ahead, ahead.conn_road,
                            ahead.road == ahead.conn_road)
                        diverging = self._negotiation.zone_map.zones(
                            own_band, lead_band)
                        if not any(z.kind == "diverge" and
                                   vehicle.position + vehicle.length / 2 -
                                   vehicle.approach_end <= z.a_out and
                                   ahead.position - ahead.length / 2 -
                                   ahead.approach_end <= z.b_out
                                   for z in diverging):
                            continue
                    seam_gap = (ahead.position - vehicle.position -
                                (ahead.length + vehicle.length) / 2.0)
                    if seam_gap < gap:
                        leader, gap = ahead, seam_gap
            vehicle.leading_distance = gap
            vehicle.leading_speed = leader.speed if leader else 0.0
            acceleration = (lane_decision.acceleration if self.mode == "gama_2025_6_4"
                            else self._idm(vehicle, gap, leader.speed if leader else 0.0))
            junction_stop = (self._junction_stop.get(vehicle.creation_id)
                             if self._negotiation is not None else None)
            if junction_stop is not None:
                # Yield to a junction foe: approach the stop point before the
                # conflict zone with the ordinary IDM (SUMO adaptToJunctionLeader).
                junction_stop, junction_obstacle_speed = junction_stop
                junction_room = junction_stop - (vehicle.position + vehicle.length / 2.0)
                acceleration = min(acceleration, self._idm(
                    vehicle, max(0.0, junction_room) + vehicle.gap, junction_obstacle_speed))
            must_hold_at_stop = (
                self.mode == "safe" and is_in_road and not committed and
                (route_invalid or denied_entry or not green or (
                    ticks_to_green_end is not None and
                    vehicle.position + max(0.0, vehicle.speed) * (DT * ticks_to_green_end) +
                    0.5 * vehicle.accel * (DT * ticks_to_green_end) ** 2 < vehicle.approach_end
                ))
            )
            if must_hold_at_stop:
                stop_gap = max(0.0, stop_position - vehicle.position)
                if stop_gap <= entry_horizon:
                    stop_accel = self._idm(vehicle, stop_gap + vehicle.gap, 0.0)
                    if stop_gap > 0.5:
                        stop_accel = max(-max(vehicle.decel * 1.4, 4.2), stop_accel)
                    acceleration = min(acceleration, stop_accel)
            initial_speed = vehicle.speed
            predicted_speed = initial_speed + acceleration * DT
            pending_entry = self.mode == "gama_2025_6_4" and not vehicle.gama_entered
            if pending_entry and predicted_speed <= 0:
                # GAMA's newSpeed==0 gate runs before on_entering_new_road and
                # register(), leaving current_road nil for this tick.
                vehicle.speed = 0.0
                survivors.append(vehicle)
                continue
            if pending_entry:
                vehicle.road = old_road
                if lane_decision.lane != vehicle.lane:
                    vehicle.lane = lane_decision.lane
                    shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                    vehicle.current_lateral_shift = shift
                    vehicle.start_lateral_shift = vehicle.target_lateral_shift = shift
                    vehicle.lane_change_progress = 1.0
                    vehicle.lane_change_from = -1
                vehicle.gama_entered = True
                self._neighbor_road_state[id(vehicle)] = (old_road, old_progress)
                spawn_entry = (old_progress, id(vehicle), vehicle)
                for sub in self._footprint(vehicle):
                    bisect.insort(lane_maps[old_road][sub], spawn_entry)
            if predicted_speed < 0:
                distance = -0.5 * initial_speed * initial_speed / acceleration
                vehicle.speed = 0.0
            else:
                distance = initial_speed * DT + 0.5 * acceleration * DT * DT
                vehicle.speed = predicted_speed
            unconstrained_distance = distance
            # The safety resolver is a final guard, not a way to brake by
            # moving a vehicle backwards. Preserve the required link clearance
            # during forward motion, including connector-to-exit approaches.
            if (self.mode == "safe" and leader is not None and math.isfinite(gap) and
                    (old_road.endswith(("_in", "_out")) or
                     leader.road.endswith("_out"))):
                clearance = max(1.0, vehicle.gap, leader.gap)
                distance = min(distance, max(0.0, gap + self._leader_advance(
                    leader, old_road, moved_ids) - clearance))
            if self.mode == "safe" and math.isfinite(gap):
                distance = min(distance, max(0.0, gap + self._leader_advance(
                    leader, old_road, moved_ids)))
            if junction_stop is not None:
                # Hard guard: the front never passes the junction stop point.
                junction_room = max(0.0, junction_stop - (vehicle.position + vehicle.length / 2.0)
                                    + junction_obstacle_speed * DT)
                if distance > junction_room + 1e-9:
                    distance = junction_room
                    vehicle.speed = max(0.0, min(vehicle.speed, 2.0 * junction_room / DT - initial_speed))
                    self._negotiation.hard_caps += 1
            if (route_invalid and vehicle.position <= stop_position <
                    vehicle.position + distance):
                distance = max(0.0, stop_position - vehicle.position)
                vehicle.speed = 0.0
            boundary = (vehicle.approach_end if is_in_road else
                        vehicle.connector_end if old_road not in
                        (f"{vehicle.to_arm}_out",) else vehicle.route_end)
            to_boundary = boundary - vehicle.position
            can_cross = ((green or safe_committed or
                          (self.mode == "safe" and vehicle.junction_entry_token)) and
                         (safe_committed or self._can_enter_lane(vehicle)) and
                         (self.mode != "safe" or committed or
                          vehicle.junction_entry_token or
                          vehicle.creation_id in self._junction_grants)
                         if is_in_road else True)
            if (self.mode == "safe" and is_in_road and vehicle.maneuver == "right"
                    and vehicle.position + vehicle.length / 2 < vehicle.approach_end):
                cannot_stop_before_line = (
                    vehicle.junction_entry_token and vehicle.speed >= 4.0 and
                    vehicle.speed ** 2 / (2.0 * max(1.0, vehicle.decel)) >=
                    max(0.0, self._stop_position(vehicle) - vehicle.position))
                can_cross = can_cross and (
                    cannot_stop_before_line or self._destination_has_room(vehicle))
            # Hold the stop-line buffer clear before a fixed phase ends or a
            # training decision can switch phase. This avoids a red-phase
            # rollback when downstream traffic blocks entry.
            future_time = (DT * max(0, ticks_to_green_end - 1)
                           if ticks_to_green_end is not None else 0.0)
            reachable_before_red = (distance + max(0.0, vehicle.speed) * future_time +
                                    0.5 * vehicle.accel * future_time * future_time)
            cannot_clear_before_red = (self.mode == "safe" and ticks_to_green_end is not None and
                not committed and
                (not can_cross or reachable_before_red < to_boundary))
            if (is_in_road and cannot_clear_before_red and
                    vehicle.position <= stop_position < vehicle.position + distance):
                distance = stop_position - vehicle.position
                vehicle.speed = 0.0
                vehicle.junction_entry_token = False
            if (can_cross and boundary < vehicle.route_end and
                    distance >= to_boundary and predicted_speed > 0):
                # GAMA reaches the node using the predicted old-road speed,
                # then enters the new road, evaluates MOBIL lane choice, and
                # computes IDM with the new road's parameters.
                remaining = max(0.0, DT - to_boundary / predicted_speed)
                pre_transition_speed = initial_speed
                pre_transition_lane = vehicle.lane
                old_entry = (old_progress, id(vehicle), vehicle)
                for sub in self._footprint(vehicle):
                    entries = lane_maps[old_road][sub]
                    index = bisect.bisect_left(entries, old_entry)
                    if index < len(entries) and entries[index] == old_entry:
                        entries.pop(index)
                if is_in_road:
                    if not self._arm_is_green(vehicle.arm):
                        if self._arm_is_yellow(vehicle.arm):
                            self.entries_during_yellow += 1
                        elif self.mode == "safe" and vehicle.maneuver == "right":
                            self.permitted_right_on_red_entries += 1
                        else:
                            self.entries_during_red += 1
                    vehicle.road = f"{vehicle.arm}_{vehicle.to_arm}"
                else:
                    vehicle.road = f"{vehicle.to_arm}_out"
                next_road = vehicle.road
                vehicle.lane = self._mapped_lane(vehicle, vehicle.lane, old_road, next_road)
                if self.mode == "gama_2025_6_4":
                    seam_progress = 0.0
                    new_shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                else:
                    seam_progress, new_shift = self._continuous_seam_pose(
                        old_road, next_road, vehicle.current_lateral_shift)
                    # Polyline seams may place the projected point slightly
                    # behind the boundary; never move the vehicle backwards.
                    seam_progress = max(0.0, seam_progress,
                                        vehicle.position - boundary)
                    vehicle.seam_from_road = old_road
                    vehicle.seam_to_road = next_road
                    vehicle.seam_progress = seam_progress
                    vehicle.seam_shift = new_shift
                vehicle.position = boundary + seam_progress
                vehicle.current_lateral_shift = new_shift
                vehicle.start_lateral_shift = new_shift
                vehicle.lane_change_from = -1
                if self.mode == "gama_2025_6_4":
                    vehicle.target_lateral_shift = new_shift
                    vehicle.lane_change_progress = 1.0
                else:
                    min_shift = ((vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                    max_shift = (self.roads[next_road]["num_lanes"] - 1 - (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                    if new_shift < min_shift - 0.05 or new_shift > max_shift + 0.05:
                        vehicle.target_lateral_shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                        vehicle.lane_change_progress = 0.0
                        vehicle.lateral_duration = max(0.8, abs(vehicle.target_lateral_shift - new_shift) / 1.8)
                    else:
                        vehicle.target_lateral_shift = new_shift
                        vehicle.lane_change_progress = 1.0
                if next_road not in lane_maps:
                    lane_maps[next_road] = self._new_road_lanes(next_road)
                self._neighbor_road_state[id(vehicle)] = (next_road, seam_progress)
                self._free_idm_cache.pop(id(vehicle), None)
                new_entry = (seam_progress, id(vehicle), vehicle)
                for sub in self._footprint(vehicle):
                    bisect.insort(lane_maps[next_road][sub], new_entry)
                if self.mode == "gama_2025_6_4":
                    # GAMA reaches the node without storing old-road's full-step
                    # predicted speed; new-road IDM starts from the stored speed.
                    vehicle.speed = pre_transition_speed
                new_lane_decision = self._choose_lane(vehicle, lane_maps)
                new_leader, new_gap, _, _ = self._neighbors(
                    vehicle, vehicle.lane, lane_maps,
                    shift=(vehicle.current_lateral_shift if self.mode == "safe" else None))
                vehicle.leading_distance = new_gap
                vehicle.leading_speed = new_leader.speed if new_leader else 0.0
                crossing_gap = new_gap if new_leader else math.inf
                new_acceleration = (new_lane_decision.acceleration
                                    if self.mode == "gama_2025_6_4" else self._idm(
                                        vehicle, crossing_gap,
                                        new_leader.speed if new_leader else 0.0))
                if self.mode == "gama_2025_6_4":
                    transition = transition_at_node(
                        speed=pre_transition_speed,
                        old_road_acceleration=acceleration,
                        distance_to_node=to_boundary,
                        remaining_time=DT,
                        dt=DT,
                        new_road_acceleration=new_acceleration,
                        current_lane=vehicle.lane,
                        num_lanes_occupied=vehicle.occupied,
                        new_road_num_lanes=self.roads[next_road]["num_lanes"])
                    if not transition.crossed:
                        vehicle.position = boundary
                        vehicle.road = old_road
                        vehicle.speed = 0.0
                        vehicle.lane = pre_transition_lane
                        new_entry = (0.0, id(vehicle), vehicle)
                        for sub in self._footprint(vehicle):
                            entries = lane_maps[next_road][sub]
                            index = bisect.bisect_left(entries, new_entry)
                            if index < len(entries) and entries[index] == new_entry:
                                entries.pop(index)
                        self._neighbor_road_state[id(vehicle)] = (old_road, old_progress)
                        restored_entry = (old_progress, id(vehicle), vehicle)
                        for sub in self._footprint(vehicle):
                            bisect.insort(lane_maps[old_road][sub], restored_entry)
                        survivors.append(vehicle)
                        continue
                    else:
                        new_distance = transition.new_road_distance
                        vehicle.speed = transition.stored_speed
                else:
                    crossing_speed = predicted_speed
                    if crossing_speed + new_acceleration * remaining < 0:
                        new_distance = -0.5 * crossing_speed * crossing_speed / new_acceleration
                        vehicle.speed = 0.0
                    else:
                        new_distance = (crossing_speed * remaining +
                                        0.5 * new_acceleration * remaining * remaining)
                        vehicle.speed = max(0.0, crossing_speed +
                                            new_acceleration * remaining)
                unconstrained_new_distance = new_distance
                if self.mode == "safe" and math.isfinite(new_gap):
                    new_distance = min(new_distance, max(0.0, new_gap))
                if (self.mode == "safe" and remaining > 0 and
                        new_distance < unconstrained_new_distance - 1e-9):
                    vehicle.speed = min(vehicle.speed, max(
                        0.0, 2.0 * new_distance / remaining - crossing_speed))
                vehicle.position += max(0.0, new_distance)
                curr_entry = (seam_progress, id(vehicle), vehicle)
                for sub in self._footprint(vehicle):
                    entries = lane_maps[next_road][sub]
                    index = bisect.bisect_left(entries, curr_entry)
                    if index < len(entries) and entries[index] == curr_entry:
                        entries.pop(index)
                final_progress = seam_progress + new_distance
                final_entry = (final_progress, id(vehicle), vehicle)
                for sub in self._footprint(vehicle):
                    bisect.insort(lane_maps[next_road][sub], final_entry)
                self._neighbor_road_state[id(vehicle)] = (next_road, final_progress)
                new_road, new_entry = next_road, final_entry
            else:
                if distance < unconstrained_distance - 1e-9:
                    vehicle.speed = min(vehicle.speed, max(
                        0.0, 2.0 * distance / DT - initial_speed))
                hold_from = vehicle.position
                vehicle.position += max(0.0, distance)
                if is_in_road and not can_cross and vehicle.position > stop_position:
                    # ``stop_position`` may be a merge buffer upstream of the
                    # line; a vehicle already beyond it holds where it is.
                    vehicle.position = max(hold_from, stop_position)
                    vehicle.speed = 0.0
                self._free_idm_cache.pop(id(vehicle), None)
                old_entry = (old_progress, id(vehicle), vehicle)
                for sub in self._footprint(vehicle):
                    entries = lane_maps[old_road][sub]
                    index = bisect.bisect_left(entries, old_entry)
                    if index < len(entries) and entries[index] == old_entry:
                        entries.pop(index)
                new_road, new_progress = self._road_state(vehicle)
                vehicle.road = new_road
                if new_road != old_road:
                    max_lane = max(0, self.roads[new_road]["num_lanes"] - vehicle.occupied)
                    if vehicle.lane > max_lane:
                        vehicle.lane = max_lane
                        new_shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                        vehicle.start_lateral_shift = vehicle.target_lateral_shift = new_shift
                        vehicle.current_lateral_shift = new_shift
                        vehicle.lane_change_progress = 1.0
                        vehicle.lane_change_from = -1
                self._neighbor_road_state[id(vehicle)] = (new_road, new_progress)
                if new_road not in lane_maps:
                    lane_maps[new_road] = self._new_road_lanes(new_road)
                new_entry = (new_progress, id(vehicle), vehicle)
                for sub in self._footprint(vehicle):
                    bisect.insort(lane_maps[new_road][sub], new_entry)
            if vehicle.position >= vehicle.route_end - 0.01:
                self.passed += 1
                # Still in ``self.vehicles`` until the tick ends; the indexed
                # neighbour queries add it back explicitly.
                self._exited_this_tick.append(vehicle)
                for sub in self._footprint(vehicle):
                    entries = lane_maps[new_road][sub]
                    index = bisect.bisect_left(entries, new_entry)
                    if index < len(entries) and entries[index] == new_entry:
                        entries.pop(index)
            else:
                min_lc_speed = 1.5 if vehicle.lane_change_from < 0 else 3.0
                road_progress = self._road_state(vehicle)[1]
                seam_clearing = (vehicle.lane_change_from < 0 and
                                 not new_road.endswith(("_in", "_out")) and
                                 road_progress < vehicle.length / 2 + 1.5)
                if vehicle.lane_change_progress < 1.0 and not seam_clearing:
                    # Below ``min_lc_speed`` the lateral move continues at a
                    # reduced rate (>= 25 %) instead of freezing: a change
                    # frozen half-way at standstill straddles two corridors
                    # and deadlocks queues (13/11 liveness audit).
                    rate = max(0.25, min(1.0, vehicle.speed / min_lc_speed))
                    if (new_road == f"{vehicle.arm}_in" and
                            (vehicle.merge_wait > 0.0 or
                             vehicle.lane != self._route_target_lane(vehicle))):
                        # Mandatory corridor merges keep their full lateral rate.
                        rate = 1.0
                    u = min(1.0, vehicle.lane_change_progress +
                            rate * DT / max(0.001, vehicle.lateral_duration))
                    # Cubic Bezier curve per Naseri et al. (2017) Section 3.1:
                    # y(u) = start + (target - start) * (3u^2 - 2u^3)
                    bezier_u = 3.0 * u * u - 2.0 * u * u * u
                    next_shift = (vehicle.start_lateral_shift +
                        (vehicle.target_lateral_shift - vehicle.start_lateral_shift) * bezier_u)
                    lo = min(vehicle.current_lateral_shift, next_shift)
                    hi = max(vehicle.current_lateral_shift, next_shift)
                    lateral_blocked = False
                    reach_long = (vehicle.length + self._max_vehicle_length) / 2 + max(
                        1.2, vehicle.gap, self._max_vehicle_gap) + 20.0 * vehicle.lateral_duration
                    nearby = list(self._nearby(new_road, road_progress - reach_long,
                                               road_progress + reach_long))
                    if new_road.endswith("_out"):
                        nearby.extend(other for other in self.vehicles
                                      if other.to_arm == vehicle.to_arm and
                                      other.road == other.conn_road and
                                      abs(other.position - other.connector_end -
                                          road_progress) <= reach_long)
                    for other in nearby:
                        if other is vehicle:
                            continue
                        other_road, other_progress = self._road_state(other)
                        other_shift = other.current_lateral_shift
                        if other_road != new_road:
                            if (not new_road.endswith("_out") or
                                    other_road != other.conn_road or
                                    other.to_arm != vehicle.to_arm):
                                continue
                            seam_progress, other_shift = self._continuous_seam_pose(
                                other.conn_road, new_road, other_shift)
                            other_progress = (other.position - other.connector_end +
                                              seam_progress)
                        separation = abs(other_progress - road_progress) - (
                            vehicle.length + other.length) / 2
                        remaining_change = max(0.0, (1.0 - vehicle.lane_change_progress)
                                               * vehicle.lateral_duration)
                        closing = max(0.0, (other.speed - vehicle.speed)
                                      if other_progress < road_progress else
                                      (vehicle.speed - other.speed))
                        moving_gap = (other.gap + min(8.0, other.speed * other.headway)
                                      if other_progress < road_progress else
                                      vehicle.gap + min(8.0, vehicle.speed * vehicle.headway))
                        if (separation - closing * remaining_change >=
                                max(1.2, vehicle.gap, other.gap, moving_gap)):
                            continue
                        if self._moving_away(vehicle.current_lateral_shift, next_shift,
                                             other_shift, other_shift):
                            continue
                        lateral_gap = max(lo - other_shift,
                                          other_shift - hi, 0.0)
                        if lateral_gap < (self._half_width(vehicle) +
                                          self._half_width(other) + LATERAL_CLEARANCE):
                            lateral_blocked = True
                            break
                    if not lateral_blocked:
                        vehicle.lane_change_progress = u
                        vehicle.current_lateral_shift = next_shift
                elif vehicle.lane_change_progress >= 1.0:
                    vehicle.current_lateral_shift = vehicle.target_lateral_shift
                survivors.append(vehicle)
        self.vehicles = survivors
        # The index is only valid inside this tick's movement pass.
        self._road_index = None
        self._exited_this_tick = []
        if self.mode == "safe":
            self._resolve_road_conflicts()
            self._resolve_connector_overlaps(starting_positions, starting_speeds)

    def _resolve_connector_overlaps(self, starting_positions, starting_speeds):
        """Stop a following connector body before it enters an occupied footprint.

        The negotiation model works with sampled centerline bands. Curved,
        offset connector lanes can still bring two rectangles together.
        This final geometry check only retreats within the distance travelled
        in the current tick and never rewinds to an earlier tick.
        """
        groups = {}
        for vehicle in self.vehicles:
            road, progress = self._road_state(vehicle)
            if not road.endswith(("_in", "_out")):
                groups.setdefault(road, []).append((progress, vehicle))
        for road, entries in groups.items():
            entries.sort(key=lambda item: item[0], reverse=True)
            placed = []
            geometry = self._geometry_roads[road]
            def make_polygon(item):
                return body_polygon(geometry.pose(
                    item.position - item.approach_end,
                    item.current_lateral_shift), length=item.length,
                    width=2 * self._half_width(item))
            for progress, vehicle in entries:
                blockers = [other for ahead, _, other in placed
                            if ahead - progress < 8.0]
                if blockers and any(polygons_overlap(make_polygon(vehicle), other, 0.0)
                                    for other in blockers):
                    end = vehicle.position
                    start = min(end, starting_positions.get(vehicle.creation_id, end))
                    vehicle.position = start
                    if not any(polygons_overlap(make_polygon(vehicle), other, 0.0)
                               for other in blockers):
                        lo, hi = start, end
                        for _ in range(18):
                            mid = (lo + hi) / 2
                            vehicle.position = mid
                            if any(polygons_overlap(make_polygon(vehicle), other, 0.0)
                                   for other in blockers):
                                hi = mid
                            else:
                                lo = mid
                        vehicle.position = lo
                    distance = vehicle.position - start
                    vehicle.speed = max(0.0, min(vehicle.speed,
                        2.0 * distance / DT - starting_speeds.get(vehicle.creation_id, 0.0)))
                placed.append((vehicle.position - vehicle.approach_end,
                               vehicle, make_polygon(vehicle)))

    def _resolve_road_conflicts(self):
        """Keep vehicle bodies and their minimum clearance apart on road links.

        Connector links form the intersection and are intentionally excluded.
        Progress is road-relative: vehicles entering the same exit from different
        arms have different cumulative route positions.
        """
        by_road = {}
        for vehicle in self.vehicles:
            road, progress = self._road_state(vehicle)
            if road.endswith(("_in", "_out")):
                by_road.setdefault(road, []).append((progress, vehicle))
        for road, entries in by_road.items():
            entries.sort(key=lambda item: item[0], reverse=True)
            settled = []
            for progress, vehicle in entries:
                half_width = self._half_width(vehicle)
                allowed = progress
                for ahead_progress, ahead in reversed(settled):
                    if ahead_progress - progress > 6.0:
                        continue
                    other_half_width = self._half_width(ahead)
                    if abs(vehicle.current_lateral_shift - ahead.current_lateral_shift) >= (
                            half_width + other_half_width + LATERAL_CLEARANCE):
                        continue
                    clearance = max(1.0, vehicle.gap, ahead.gap)
                    allowed = min(allowed, ahead_progress -
                                  (vehicle.length + ahead.length) / 2 - clearance)
                    if allowed < progress:
                        vehicle.speed = min(vehicle.speed, ahead.speed)
                if allowed < progress:
                    correction = progress - allowed
                    vehicle.position -= correction
                    # Hold a vehicle at the end of its connector when the exit
                    # lacks room. Keep the road state consistent with its
                    # position so the next step can wait for the exit to clear.
                    if road.endswith("_out") and allowed < 0.0:
                        connector = f"{vehicle.arm}_{vehicle.to_arm}"
                        vehicle.lane = self._mapped_lane(
                            vehicle, vehicle.lane, road, connector)
                        shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                        vehicle.road = connector
                        vehicle.current_lateral_shift = shift
                        vehicle.start_lateral_shift = shift
                        vehicle.target_lateral_shift = shift
                        vehicle.lane_change_progress = 1.0
                        vehicle.lane_change_from = -1
                    self.road_safety_corrections += 1
                    if correction > self.max_road_safety_correction:
                        self.largest_road_safety_event = (
                            self.cycle, road, vehicle.demand_row_id, correction)
                    self.max_road_safety_correction = max(
                        self.max_road_safety_correction, correction)
                if vehicle.road == road:
                    settled.append((allowed, vehicle))

    def _choose_lane(self, vehicle, lane_maps):
        if self.mode == "gama_2025_6_4":
            return self._choose_lane_gama(vehicle, lane_maps)
        road, progress = self._neighbor_road_state[id(vehicle)]
        if vehicle.lane_change_progress < 1.0:
            vehicle.time_since_lane_change += DT
            return
        if not road.endswith(("_in", "_out")):
            vehicle.time_since_lane_change += DT
            return
        stop_gap = (self._stop_position(vehicle) - vehicle.position
                    if road.endswith("_in") else math.inf)
        green = self._arm_is_green(vehicle.arm)
        # Slow motorcycles may filter through a red-light queue while there
        # is still room to finish the lateral movement before the stop line.
        queue_filter = (road.endswith("_in") and vehicle.occupied == 1 and
                        not green and 1.5 <= vehicle.speed <= 5.0 and
                        max(4.0, vehicle.speed * 1.1 + 1.0) < stop_gap <= 35.0)
        turn_target = self._route_target_lane(vehicle)
        mandatory = road.endswith("_in") and vehicle.lane != turn_target
        if mandatory and vehicle.merge_wait >= MERGE_PATIENCE:
            step = 1 if turn_target > vehicle.lane else -1
            vehicle.merge_claim_shift = (vehicle.lane + step +
                                         (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
        else:
            vehicle.merge_claim_shift = math.nan
        if vehicle.speed < 3.0 and not queue_filter and not mandatory:
            vehicle.time_since_lane_change += DT
            return
        if (road.endswith("_in") and not queue_filter and not mandatory and stop_gap <=
                max(20.0, vehicle.speed * (vehicle.lateral_duration + DT) + 5.0)):
            vehicle.time_since_lane_change += DT
            return
        old_leader, old_gap, old_follower, _ = self._neighbors(
            vehicle, vehicle.lane, lane_maps)
        overlap = old_gap < 0
        if (not mandatory and not queue_filter and not overlap and
                old_gap < vehicle.gap + vehicle.speed * vehicle.headway):
            vehicle.time_since_lane_change += DT
            return
        if not overlap and not mandatory and not queue_filter:
            if (old_gap > max(25.0, vehicle.speed * 3.0) or
                    old_leader is None or old_leader.speed >= 0.92 * vehicle.desired or
                    (road.endswith("_in") and vehicle.maneuver != "through" and
                     self._stop_position(vehicle) - vehicle.position <= 40.0)):
                vehicle.time_since_lane_change += DT
                vehicle.leading_distance = old_gap
                vehicle.leading_speed = old_leader.speed if old_leader else 0.0
                return
        if not overlap and vehicle.time_since_lane_change < (0.8 if mandatory or queue_filter else 4.0):
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
        min_gain = 0.5 if mandatory else (1.5 if queue_filter else 0.8)
        for lane in range(self.roads[road]["num_lanes"] - vehicle.occupied + 1):
            if abs(lane - vehicle.lane) != 1:
                continue
            if road.endswith("_in"):
                old_lane = vehicle.lane
                vehicle.lane = lane
                route_valid = self._route_target_lane(vehicle) == lane
                vehicle.lane = old_lane
                if not route_valid and not mandatory:
                    continue
            if mandatory and abs(lane - turn_target) >= abs(vehicle.lane - turn_target):
                continue
            leader, front_gap, follower, back_gap = self._neighbors(vehicle, lane, lane_maps)
            front_required = (vehicle.gap + vehicle.speed * vehicle.headway +
                              (max(0.0, (vehicle.speed - leader.speed) * 1.2)
                               if leader is not None else 0.0))
            back_required = (follower.gap + follower.speed * follower.headway +
                             max(0.0, (follower.speed - vehicle.speed) * 1.5 + 1.0)
                             if follower is not None else 0.0)
            if mandatory and vehicle.merge_wait > 0.0:
                urgency = min(1.0, vehicle.merge_wait / MERGE_PATIENCE)
                front_required = ((1.0 - urgency) * front_required +
                                  urgency * vehicle.gap)
                if follower is not None:
                    # Minimum: the follower can still stop behind the merger.
                    # 5 cm tolerance: a courteous follower stops exactly at
                    # its jam gap, which must not count as "too close".
                    back_floor = follower.gap - 0.05 + follower.speed ** 2 / (
                        2.0 * max(4.0, 1.5 * follower.decel))
                    back_required = ((1.0 - urgency) * back_required +
                                     urgency * back_floor)
            if front_gap < front_required or back_gap < back_required:
                continue
            if queue_filter and (min(front_gap, stop_gap) - min(old_gap, stop_gap) <= 1.5):
                continue
            cand_shift = (lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
            urgency = (min(1.0, vehicle.merge_wait / MERGE_PATIENCE)
                       if mandatory else 0.0)
            lo_lat = min(vehicle.current_lateral_shift, cand_shift)
            hi_lat = max(vehicle.current_lateral_shift, cand_shift)
            sweep_conflict = False
            reach_long = (vehicle.length + self._max_vehicle_length) / 2 + max(
                1.5, vehicle.gap, self._max_vehicle_gap) + 20.0 * max(vehicle.lateral_duration, DT)
            for other in self._nearby(road, progress - reach_long, progress + reach_long):
                if other is vehicle:
                    continue
                other_road, other_pos = self._neighbor_road_state.get(id(other), ("", 0.0))
                if other_road != road:
                    continue
                long_sep = abs(other_pos - progress) - (vehicle.length + other.length) / 2
                closing = max(0.0, (other.speed - vehicle.speed)
                              if other_pos < progress else
                              (vehicle.speed - other.speed))
                long_sep -= closing * max(vehicle.lateral_duration, DT)
                # Impatient mergers accept the smaller of the two jam gaps.
                if long_sep >= ((1.0 - urgency) * max(1.5, vehicle.gap, other.gap) +
                                urgency * (min(vehicle.gap, other.gap) - 0.05)):
                    continue
                other_lo = min(other.current_lateral_shift, other.target_lateral_shift)
                other_hi = max(other.current_lateral_shift, other.target_lateral_shift)
                if self._moving_away(vehicle.current_lateral_shift, cand_shift,
                                     other_lo, other_hi):
                    continue
                lat_gap = max(lo_lat - other_hi, other_lo - hi_lat, 0.0)
                if lat_gap < self._half_width(vehicle) + self._half_width(other) + LATERAL_CLEARANCE:
                    sweep_conflict = True
                    break
            if sweep_conflict:
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
                    if after <= -2.5:
                        continue
                    before = self._idm(follower, prior_gap, leader_speed)
                    follower_delta = after - before
            gain = (min(front_gap, stop_gap) - min(old_gap, stop_gap)
                    if queue_filter else target_acc - old_acc + 0.1 * follower_delta)
            if mandatory:
                gain += 2.0 + max(0.0, 40.0 - stop_gap) * 0.15
                if vehicle.merge_wait > 0.0:
                    # A waiting merger takes any safe gap closer to its corridor.
                    gain = max(gain, min_gain + 1e-3)
            if gain > max(min_gain, best_gain):
                best_lane, best_gain = lane, gain
                best_leader, best_front_gap = leader, front_gap
        if best_lane != vehicle.lane:
            vehicle.lane_change_from = vehicle.lane
            entry = (progress, id(vehicle), vehicle)
            for sub in self._footprint(vehicle):
                entries = lane_maps[road][sub]
                index = bisect.bisect_left(entries, entry)
                if index < len(entries) and entries[index] == entry:
                    entries.pop(index)
            vehicle.lane = best_lane
            vehicle.lane_change_progress = 0.0
            for sub in self._footprint(vehicle):
                bisect.insort(lane_maps[road][sub], entry)
            vehicle.time_since_lane_change = 0.0
            vehicle.leading_distance = best_front_gap
            vehicle.leading_speed = best_leader.speed if best_leader else 0.0
            # Smooth Bézier lateral transition per Naseri et al. (2017) Section 3.1
            vehicle.start_lateral_shift = vehicle.current_lateral_shift
            vehicle.target_lateral_shift = (best_lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
            vehicle.lane_change_progress = 0.0
            lateral_dist = abs(vehicle.target_lateral_shift - vehicle.start_lateral_shift)
            # Paper Eq: a_allow = 2.0 m/s^2, r = v^2 / a_allow, T_lc = sqrt(6 * W / a_allow)
            a_allow = 2.0
            base_duration = math.sqrt(6.0 * max(0.4, lateral_dist) / a_allow)
            multiplier = 0.6 if vehicle.occupied == 1 else 1.0
            vehicle.lateral_duration = max(0.4, base_duration * multiplier)
        else:
            vehicle.time_since_lane_change += DT

    def _gama_idm_params(self, vehicle):
        return GamaIDMParams(
            time_headway=vehicle.headway,
            max_acceleration=vehicle.accel,
            max_deceleration=vehicle.decel,
            desired_speed=vehicle.desired,
            min_safety_distance=vehicle.gap,
            delta=4.0,
            speed_coeff=1.0)

    def _gama_road_limit(self, road):
        return (40.0 if self.roads[road]["kind"] == "connector" else 60.0) / 3.6

    def _choose_lane_gama(self, vehicle, lane_maps):
        """Apply tagged GAMA 2025.6.4 IDM/MOBIL rules on current lane maps."""
        road, progress = self._neighbor_road_state[id(vehicle)]
        current_leader, current_gap, current_follower, current_back_gap = self._neighbors(
            vehicle, vehicle.lane, lane_maps)
        # Utils.findLeader/findFollower use 1e6 m when there is no neighbor.
        current_gap = 1e6 if current_gap == math.inf else current_gap
        old_acc = (-sys.float_info.max if current_gap < 0 else gama_idm_acceleration(
            speed=vehicle.speed,
            leading_speed=current_leader.speed if current_leader else 0.0,
            gap=current_gap, params=self._gama_idm_params(vehicle),
            road_max_speed=self._gama_road_limit(road)))
        green = ((vehicle.arm in ("NE", "SW") and self.phase == "LVL") or
                 (vehicle.arm in ("SE", "NW") and self.phase == "HDT"))
        candidates = {}
        upper = self.roads[road]["num_lanes"] - vehicle.occupied
        lower = max(0, vehicle.lane - 1)
        upper = min(upper, vehicle.lane + 1)
        for lane in range(lower, upper + 1):
            if lane == vehicle.lane:
                continue
            leader, front_gap, follower, back_gap = self._neighbors(vehicle, lane, lane_maps)
            candidates[lane] = GamaCandidate(
                front_gap=1e6 if front_gap == math.inf else front_gap,
                leader_speed=leader.speed if leader else 0.0,
                follower_id=id(follower) if follower else None,
                back_gap=1e6 if back_gap == math.inf else back_gap,
                follower_speed=follower.speed if follower else 0.0,
                follower_leading_distance=(follower.leading_distance if follower else math.inf),
                follower_params=self._gama_idm_params(follower) if follower else self._gama_idm_params(vehicle),
                follower_road_max_speed=self._gama_road_limit(road))
        result = gama_choose_lane(
            current_lane=vehicle.lane,
            lane_count=self.roads[road]["num_lanes"],
            lanes_occupied=vehicle.occupied,
            speed=vehicle.speed,
            vehicle_length=vehicle.length,
            current_gap=current_gap,
            current_leader_speed=current_leader.speed if current_leader else 0.0,
            current_follower_id=id(current_follower) if current_follower else None,
            current_time_since_lane_change=vehicle.time_since_lane_change,
            timestep=DT,
            vehicle_params=self._gama_idm_params(vehicle),
            candidates=candidates,
            mobil=GamaMobilParams(
                politeness=0.1, safe_deceleration=4.0,
                acceleration_threshold=0.5, acceleration_bias=0.25,
                right_side_driving=True, lane_change_limit=1, cooldown=4.0),
            road_max_speed=self._gama_road_limit(road))
        vehicle.time_since_lane_change = result.time_since_lane_change
        if result.lane != vehicle.lane and vehicle.gama_entered:
            old_lane = vehicle.lane
            entry = (progress, id(vehicle), vehicle)
            for sub in range(old_lane, old_lane + vehicle.occupied):
                entries = lane_maps[road][sub]
                index = bisect.bisect_left(entries, entry)
                if index < len(entries) and entries[index] == entry:
                    entries.pop(index)
            vehicle.lane = result.lane
            if vehicle.gama_entered:
                for sub in range(vehicle.lane, vehicle.lane + vehicle.occupied):
                    bisect.insort(lane_maps[road][sub], entry)
            vehicle.lane_change_from = old_lane
            vehicle.lane_change_progress = 1.0
            vehicle.current_lateral_shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
            vehicle.start_lateral_shift = vehicle.target_lateral_shift = vehicle.current_lateral_shift
        selected_lane = result.lane
        new_leader, new_gap, _, _ = self._neighbors(vehicle, selected_lane, lane_maps)
        vehicle.leading_distance = 1e6 if new_gap == math.inf else new_gap
        vehicle.leading_speed = new_leader.speed if new_leader else 0.0
        return result

    def _road_state(self, vehicle):
        # Road names are precomputed per vehicle (``Vehicle.__post_init__``);
        # this is the hottest function of the simulator.
        road = vehicle.road
        in_road, conn_road = vehicle.in_road, vehicle.conn_road
        if road:
            if road == in_road and vehicle.position > vehicle.approach_end:
                road = vehicle.road = conn_road
            elif road == conn_road and vehicle.position > vehicle.connector_end:
                road = vehicle.road = vehicle.out_road
        if not road:
            if vehicle.position <= vehicle.approach_end:
                road = vehicle.road = in_road
            elif vehicle.position <= vehicle.connector_end:
                road = vehicle.road = conn_road
            else:
                road = vehicle.road = vehicle.out_road
        if road == in_road:
            return road, vehicle.position
        if road == conn_road:
            return road, vehicle.position - vehicle.approach_end
        return road, vehicle.position - vehicle.connector_end

    @staticmethod
    def _overlap(a, a_lane, b):
        return a_lane < b.lane + b.occupied and b.lane < a_lane + a.occupied

    def _neighbors(self, vehicle, lane, lane_maps, shift=None):
        road, position = self._neighbor_road_state[id(vehicle)]
        candidate_shift = ((lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                           if shift is None else shift)
        def lateral_interacts(other):
            if self.mode != "safe":
                return True
            reach = self._half_width(vehicle) + self._half_width(other) + LATERAL_CLEARANCE
            return (abs(candidate_shift - other.current_lateral_shift) < reach or
                    abs(candidate_shift - other.merge_claim_shift) < reach)
        leader = follower = None
        front = back = math.inf
        front_distance = back_distance = math.inf
        for sublane in range(lane, lane + vehicle.occupied):
            entries = lane_maps[road][sublane]
            index = bisect.bisect_left(entries, (position, -1))
            ahead = index
            while ahead < len(entries) and (entries[ahead][2] is vehicle or
                                            not lateral_interacts(entries[ahead][2])):
                ahead += 1
            if ahead < len(entries):
                other_position, _, other = entries[ahead]
                distance = other_position - position
                if distance < front_distance:
                    front_distance = distance
                    leader = other
                    front = distance - (vehicle.length + other.length) / 2
            behind = index - 1
            while behind >= 0 and (entries[behind][2] is vehicle or
                                   not lateral_interacts(entries[behind][2])):
                behind -= 1
            if behind >= 0:
                other_position, _, other = entries[behind]
                distance = position - other_position
                if distance < back_distance:
                    back_distance = distance
                    follower = other
                    back = distance - (vehicle.length + other.length) / 2
        if self.mode == "safe":
            # ``shift`` lets the caller query the body's actual lateral
            # position (mid lane change / after a seam) instead of the sublane
            # centre; the resolver and the lateral checks use that position.
            candidate_shift = ((lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                               if shift is None else shift)
            half_width = self._half_width(vehicle)

            def consider(other):
                # Returns the live road-relative delta when ``other`` interacts
                # with the candidate sublane, else None (original predicate).
                other_road, other_position = self._road_state(other)
                if other_road != road:
                    return None
                reach = half_width + self._half_width(other) + LATERAL_CLEARANCE
                if abs(candidate_shift - other.current_lateral_shift) >= reach:
                    # Courtesy: a claimed merge ahead counts as a leader.
                    # Only when the merger is fully ahead; a vehicle already
                    # alongside keeps going so the merger can slot in behind.
                    if not (other_position - position >= (vehicle.length + other.length) / 2
                            and abs(candidate_shift - other.merge_claim_shift) < reach):
                        return None
                return other_position - position

            # Leader/follower = smallest bumper gap (not centre distance):
            # a short motorbike whose centre is nearer must not hide a longer
            # car whose rear is closer (11/11 audit, _neighbors).
            reach_len = (vehicle.length + self._max_vehicle_length) / 2

            def offer(other, delta):
                nonlocal leader, front, follower, back
                if delta >= 0:
                    gap = delta - (vehicle.length + other.length) / 2
                    if gap < front:
                        leader, front = other, gap
                else:
                    gap = -delta - (vehicle.length + other.length) / 2
                    if gap < back:
                        follower, back = other, gap

            road_index = (self._road_index.get(road)
                          if getattr(self, "_road_index", None) is not None else None)
            if road_index is None:
                pool = self.vehicles
            else:
                # Outward scans in progress order; stop once even the longest
                # vehicle could not beat the best gap found (+ margin).
                entries = road_index.entries
                start = bisect.bisect_left(entries, (position - INDEX_MARGIN, -1))
                for k in range(start, len(entries)):
                    p_other, _, other = entries[k]
                    if p_other - position - reach_len >= front + INDEX_MARGIN:
                        break
                    if other is not vehicle:
                        delta = consider(other)
                        if delta is not None:
                            offer(other, delta)
                for k in range(start - 1, -1, -1):
                    p_other, _, other = entries[k]
                    if position - p_other - reach_len >= back + INDEX_MARGIN:
                        break
                    if other is not vehicle:
                        delta = consider(other)
                        if delta is not None:
                            offer(other, delta)
                pool = self._exited_this_tick
            for other in pool:
                if other is not vehicle:
                    delta = consider(other)
                    if delta is not None:
                        offer(other, delta)
            if leader is not None:
                front_distance = front + (vehicle.length + leader.length) / 2
            if follower is not None:
                back_distance = back + (vehicle.length + follower.length) / 2
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
                green = self._green_for(vehicle)
                if green or (self.mode == "gama_2025_6_4" and not vehicle.gama_entered):
                    next_road = f"{vehicle.arm}_{vehicle.to_arm}"
                else:
                    if self.mode == "gama_2025_6_4" and vehicle.gama_entered:
                        front = vehicle.approach_end - vehicle.position + vehicle.gap
                    elif self.mode == "safe":
                        front = self._stop_position(vehicle) - vehicle.position + vehicle.gap
            elif road == f"{vehicle.arm}_{vehicle.to_arm}":
                next_road = f"{vehicle.to_arm}_out"
            elif (self.mode == "gama_2025_6_4" and
                  road == f"{vehicle.to_arm}_out"):
                # Tagged Utils.findLeader returns a virtual target-node leader
                # when the vehicle has no next road (the final exit segment).
                front = self.roads[road]["length"] - position + vehicle.gap
            if next_road in lane_maps:
                next_lane = self._mapped_lane(vehicle, lane, road, next_road)
                seam_shift = (self._continuous_seam_pose(
                    road, next_road, vehicle.current_lateral_shift)[1]
                    if self.mode == "safe" else 0.0)
                for sublane in range(next_lane, next_lane + vehicle.occupied):
                    entries = lane_maps[next_road][sublane]
                    for other_position, _, other in entries:
                        if self.mode == "safe":
                            reach = (self._half_width(vehicle) +
                                     self._half_width(other) + LATERAL_CLEARANCE)
                            claim_interacts = (math.isfinite(other.merge_claim_shift) and
                                               abs(seam_shift - other.merge_claim_shift) < reach)
                            if (abs(seam_shift - other.current_lateral_shift) >= reach and
                                    not claim_interacts):
                                continue
                        distance = (self.roads[road]["length"] - position +
                                    other_position)
                        if distance < front_distance:
                            front_distance = distance
                            leader = other
                            front = distance - (vehicle.length + other.length) / 2
                        break
        return leader, front, follower, back

    def _idm(self, vehicle, gap, leader_speed):
        if self.mode == "gama_2025_6_4":
            road, _ = self._neighbor_road_state.get(id(vehicle), (vehicle.road, 0.0))
            if gap == math.inf:
                gap = 1e6
            return gama_idm_acceleration(
                speed=vehicle.speed, leading_speed=leader_speed, gap=gap,
                params=self._gama_idm_params(vehicle),
                road_max_speed=self._gama_road_limit(road))
        if gap == math.inf:
            cached = self._free_idm_cache.get(id(vehicle))
            if cached is not None:
                return cached
        road_limit = (40.0 if vehicle.approach_end <= vehicle.position <
                      vehicle.connector_end else 60.0) / 3.6
        limit = min(vehicle.desired, road_limit)
        if vehicle.position < vehicle.approach_end:
            green = self._green_for(vehicle)
            if ((not green and vehicle.position <= self._stop_position(vehicle) + 1e-8)
                    or self._route_target_lane(vehicle) != vehicle.lane):
                stop_gap = self._stop_position(vehicle) - vehicle.position + vehicle.gap
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
        used = {arm: set() for arm in ARMS}
        for v in self.vehicles:
            if ((v.road == v.in_road or not v.road)
                    and 0 <= v.approach_end - v.position <= 1.5):
                used[v.arm].update(range(v.lane, v.lane + v.occupied))
        for arm in ARMS:
            self.front_row_max[arm] = max(self.front_row_max[arm], len(used[arm]))

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
                  "queue_time_integral": round(self.queue_time_integral, 3),
                  "phase_remaining_sec": round(self._phase_remaining(), 2)}
        result.update(system_time_integral=round(self.system_time_integral, 3),
                      weighted_delay_integral=round(self.weighted_delay_integral, 3),
                      dqn_waiting_action=self.dqn_waiting_action,
                      next_decision_in_cycles=self.next_decision_in_cycles)
        result.update({f"front_row_max_{arm}": self.front_row_max[arm] for arm in ARMS})
        result.update(yellow_phase=self.yellow_phase or "",
                      entries_during_yellow=self.entries_during_yellow,
                      entries_during_red=self.entries_during_red,
                      permitted_right_on_red_entries=self.permitted_right_on_red_entries)
        return result

    def _phase_remaining(self):
        if self.train_mode:
            ticks = self.yellow_ticks_left if self.yellow_ticks_left > 0 else self.red_ticks_left
            if ticks > 0:
                return ticks * DT
            return max(0, self.next_decision_in_cycles) * DT
        t = max(0, self.cycle - 1) * DT
        length = self.green_lvl + self.green_hdt + 2 * self.all_red
        if length <= 0:
            return 0.0
        t %= length
        hdt_start = self.green_lvl + self.all_red
        if self.yellow_phase == "LVL":
            boundary = self.green_lvl
        elif self.yellow_phase == "HDT":
            boundary = hdt_start + self.green_hdt
        elif self.phase == "LVL":
            boundary = self.green_lvl - self._yellow
        elif self.phase == "HDT":
            boundary = hdt_start + self.green_hdt - self._yellow
        elif t < self.green_lvl:
            boundary = self.green_lvl
        elif t < hdt_start:
            boundary = hdt_start
        elif t < hdt_start + self.green_hdt:
            boundary = hdt_start + self.green_hdt
        else:
            boundary = length
        return max(0.0, boundary - t)


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
    parser.add_argument("--all-red", type=float, default=2)
    parser.add_argument("--yellow", type=float, default=4.0,
                        help="yellow interval carved from the end of each green")
    parser.add_argument("--demand-file", type=Path, default=None,
                        help="Replay demand CSV file (recommended for GAMA parity)")
    parser.add_argument("--mode", choices=("safe", "gama_2025_6_4"), default="safe",
                        help="safe simulation (default) or GAMA 2025.6.4 behavior profile")
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
        yellow=args.yellow,
        demand_file=args.demand_file, mode=args.mode)
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
