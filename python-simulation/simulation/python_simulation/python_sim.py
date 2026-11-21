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
from dataclasses import fields
from copyreg import __newobj__
from functools import cached_property, lru_cache
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
    from .vehicle_geometry import Pose
except ImportError:
    from vehicle_geometry import Road as GeometryRoad, SUBLANE_WIDTH
    from junction_conflicts import ConflictEngine, ConflictMap, Movement, MovementRequest
    from junction_negotiation import SimulatorAdapter
    from vehicle_geometry import footprint as body_polygon, polygons_overlap
    from vehicle_geometry import Pose


if __package__:
    from ._native import ROOT
else:
    ROOT = Path(__file__).resolve().parent
ARMS = ("NE", "SE", "SW", "NW")
DT = 0.2
SEAM_FOLLOW_DISTANCE = 40.0
STOP_MARGIN = 0.5
LATERAL_CLEARANCE = 0.1
# A stationary entry grant older than this is withdrawn and re-requested.
TOKEN_STALL_LIMIT = 2.0
TOKEN_STALL_COOLDOWN = 1.0
# Seconds a vehicle waiting to merge into its route corridor needs before it
# accepts the minimum physical gap (forced/courtesy merge, cf. SUMO
# lcAssertive/impatience). Behavioural assumption, not calibrated.
MERGE_PATIENCE = 8.0


@lru_cache(maxsize=256)
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


class _SeamFollowIndex:
    """Candidates per approach and 40 m query cell, in simulation order."""

    def __init__(self, vehicles):
        self.by_arm = {}
        for order, vehicle in enumerate(vehicles):
            if vehicle.road not in (vehicle.in_road, vehicle.conn_road):
                continue
            cells = self.by_arm.setdefault(vehicle.arm, {})
            cell = math.floor(vehicle.position / SEAM_FOLLOW_DISTANCE)
            # A follower in either of these cells can see this vehicle.
            cells.setdefault(cell - 1, []).append(order)
            cells.setdefault(cell, []).append(order)

    def update(self, order, vehicle, old_cell):
        new_cell = (math.floor(vehicle.position / SEAM_FOLLOW_DISTANCE)
                    if vehicle.road != vehicle.out_road else None)
        if new_cell == old_cell:
            return
        cells = self.by_arm[vehicle.arm]
        for cell in (old_cell - 1, old_cell):
            cells[cell].remove(order)
        if new_cell is not None:
            for cell in (new_cell - 1, new_cell):
                bisect.insort(cells.setdefault(cell, []), order)


def _replace_sorted(entries, old_entry, new_entry):
    """Move one existing entry, usually without passing another vehicle."""
    index = bisect.bisect_left(entries, old_entry)
    if ((index == 0 or entries[index - 1] < new_entry) and
            (index + 1 == len(entries) or new_entry < entries[index + 1])):
        entries[index] = new_entry
    else:
        del entries[index]
        list.insert(entries, bisect.bisect_left(entries, new_entry), new_entry)


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

    def move(self, lanes, footprint, old_entry, new_entry):
        if old_entry[0] == new_entry[0]:
            return
        for sublane in footprint:
            _replace_sorted(lanes[sublane], old_entry, new_entry)
        _replace_sorted(self.entries, old_entry, new_entry)
        self.counts[new_entry[:2]] = self.counts.pop(old_entry[:2])

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
        entry = super().pop(index)
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
    # "Bus & LCV" class of the same table (z = -1.55, p = 0.121: the paper
    # keeps it although it is not significant at the 5 % level).
    "heavy": -1.471,
}
# Above this deceleration a "stop" is physically infeasible -> the vehicle goes.
# (Chauhan et al. report decelerations down to about -6 m/s^2 in the amber.)
STOP_GO_MAX_DECEL = 6.0
# Beyond this distance from the stop line at yellow the vehicle simply stops.
STOP_GO_MAX_DISTANCE = 150.0

# Vehicle types. The index is the type code of the demand file (third
# column): 0 motorbike, 1 car, 2 heavy vehicle.
VEHICLE_KINDS = ("motorbike", "car", "heavy")

# Parameters per vehicle type: car following (IDM: a, b, s0, T), body size
# (m) and desired speed (km/h, normal distribution). The car and motorbike
# values are the model's original engineering values, not calibrated;
# override any of them with ``vehicle_params`` (see calibration.py).
#
# "heavy" merges trucks and buses/coaches into one class. Neither GAMA nor
# SUMO has such a merged class, so the values combine their defaults:
#   length, width, accel: SUMO vClass defaults (SUMOVehicleClass.cpp
#     getDefaultVehicleLength, SUMOVTypeParameter.cpp VClassDefaultValues /
#     getDefaultAccel) for "truck" (7.1 m, 2.4 m, 1.3 m/s^2) and "bus"
#     (12 m, 2.5 m, 1.2 m/s^2), weighted 59:41 like small_truck_bus :
#     heavy_truck_bus in the Nguyen Trai - Khuat Duy Tien counts
#     (train/benchmarks/gama/data/traffic_data_NT_KDT.csv).
#   decel: SUMO getDefaultDecel gives trucks/buses 4.0 against 4.5 for
#     passenger cars; the same ratio applied to this model's car value.
#   gap: SUMO minGap is the same (2.5 m) for cars, trucks and buses, so the
#     model's car value is kept.
#   headway: default ``time_headway`` of GAMA's driving skill (1.5 s,
#     DrivingSkill.java). SUMO uses one tau for every class; heavy vehicles
#     keeping more headway than cars is the modelling choice here.
#   speed: SUMO speedFactor has mean 1 for every class and deviation 0.05
#     for trucks against 0.1 for cars: the car mean with half its spread.
#   axles: the body is the AASHTO single-unit truck design vehicle SU (Green
#     Book, metric: 9.2 x 2.4 m, front overhang 1.2 m, rear overhang 1.8 m,
#     wheelbase 6.1 m). With the 9.0 m length kept: front overhang 1.2 m,
#     rear overhang 1.7 m, wheelbase 6.1 m.
#   min_turn_radius: AASHTO SU-30 minimum centreline turning radius (front
#     axle centre at steering lock), 38 ft = 11.6 m.
#   lat_accel: lateral acceleration in curves; urban bus passengers rate
#     1.5 m/s^2 "uncomfortable" (Nguyen et al., Singapore). Light vehicles
#     use the simulator-wide ``curve_lat_accel``.
# The width fixes the number of sublanes a type occupies (GAMA
# ``num_lanes_occupied``): ceil(width / SUBLANE_WIDTH) = 1, 2 and 3.
DEFAULT_VEHICLE_PARAMS = {
    "car": {"accel": 2.5, "decel": 3.0, "gap": 1.0, "headway": 1.2,
            "length": 4.5, "width": 1.7, "speed_mean": 50.0, "speed_sd": 5.0},
    "motorbike": {"accel": 2.0, "decel": 3.0, "gap": 0.3, "headway": 0.6,
                  "length": 1.9, "width": 0.7, "speed_mean": 40.0, "speed_sd": 5.0},
    "heavy": {"accel": 1.25, "decel": 2.7, "gap": 1.0, "headway": 1.5,
              "length": 9.0, "width": 2.45, "speed_mean": 50.0, "speed_sd": 2.5,
              "front_overhang": 1.2, "rear_overhang": 1.7,
              "min_turn_radius": 11.6, "lat_accel": 1.5},
}
# Share of heavy vehicles in the generated demand. With the model's 85 %
# motorbikes, the remaining 15 % split like car : (truck + bus) in the same
# counts (22.5 % : 4.9 %) gives 2.7 %. The counts are from another junction:
# an assumption for Le Van Luong - Hoang Dao Thuy, to be replaced by counts.
DEFAULT_HEAVY_RATIO = 0.03
# Body widths (m) by number of occupied sublanes, used where only a lane
# movement id ("<connector>@<lane>:<occupied>") is known.
HALF_WIDTH_BY_OCCUPIED = {
    math.ceil(values["width"] / SUBLANE_WIDTH - 1e-9): values["width"] / 2.0
    for values in DEFAULT_VEHICLE_PARAMS.values()
}
# Reference length of the longest light vehicle. Distance windows that were
# tuned for it grow by the excess length of longer vehicles.
LIGHT_VEHICLE_LENGTH = 4.5


def vehicle_kind(value):
    """Normalise a vehicle type given as name, demand code or car flag."""
    if isinstance(value, str):
        if value in VEHICLE_KINDS:
            return value
    elif isinstance(value, bool):
        return "car" if value else "motorbike"
    elif isinstance(value, int) and 0 <= value < len(VEHICLE_KINDS):
        return VEHICLE_KINDS[value]
    raise ValueError(f"unknown vehicle type {value!r}")


def kind_from_draw(draw, motorbike_ratio, heavy_ratio):
    """Vehicle type from one uniform draw.

    Heavy vehicles take the low end of the interval that used to be cars, so
    ``heavy_ratio = 0`` reproduces the two-type demand draw for draw.
    """
    if draw < heavy_ratio:
        return "heavy"
    return "car" if draw < 1.0 - motorbike_ratio else "motorbike"


def occupied_sublanes(width):
    return max(1, math.ceil(width / SUBLANE_WIDTH - 1e-9))

# Lateral movement. A sublane change lasts sqrt(6 W / a_allow) (Naseri et al.
# 2017) times this factor; the second table is the duration (s) a vehicle
# starts with. Motorbikes move sideways faster than cars. Heavy vehicles are
# slower: an assumption, neither GAMA nor SUMO sets this per vehicle class.
LATERAL_DURATION_FACTOR = {"motorbike": 0.6, "car": 1.0, "heavy": 1.3}
LATERAL_DURATION = {"motorbike": 0.6, "car": 1.2, "heavy": 1.6}

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


try:
    import cython
except ImportError:
    if __package__:
        from ._native import cython
    else:
        from _native import cython

if cython.compiled:
    from dataclasses import dataclass as vehicle_dataclass
else:
    from dataclasses import dataclass
    vehicle_dataclass = dataclass(slots=True)


@vehicle_dataclass
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
    # Seconds standing still on the connector (jam safety net).
    junction_standstill: float = 0.0
    # Lateral evasive manoeuvre in the junction (cut tail / run in front):
    # until it completes, the junction negotiation treats the vehicle as
    # already in its target band.
    evading: bool = False
    # Vehicle type ("motorbike", "car", "heavy") and half body width (m).
    # Left empty they follow from the number of occupied sublanes.
    kind: str = ""
    half_width: float = 0.0
    # Sideways reach (m) of the body beyond its path band where the path
    # bends under a long rigid body (see ``_update_body_sweep``).
    body_sweep: float = 0.0
    # Two-axle rigid body (heavy vehicles; 0 = body tangent to its path at
    # its centre). The front axle follows the sublane path, the rear axle
    # trails it (tractrix): see ``_axles``.
    wheelbase: float = 0.0
    front_overhang: float = 0.0
    leading_heavy: bool = False
    leading_from: tuple = ("", 0.0)
    rear_axle: tuple | None = None
    front_axle_s: float = math.nan
    # Lateral acceleration allowed in curves (m/s^2); 0 = simulator value.
    lat_accel: float = 0.0

    in_road: str = ""
    conn_road: str = ""
    out_road: str = ""

    def __reduce_ex__(self, protocol):
        state = {field.name: getattr(self, field.name) for field in fields(self)}
        return __newobj__, (type(self),), (None, state)

    def __setstate__(self, state):
        for name, value in state[1].items():
            setattr(self, name, value)

    def __post_init__(self):
        self.in_road = f"{self.arm}_in"
        self.conn_road = f"{self.arm}_{self.to_arm}"
        self.out_road = f"{self.to_arm}_out"
        if not self.kind:
            self.kind = VEHICLE_KINDS[min(self.occupied, len(VEHICLE_KINDS)) - 1]
        if self.half_width <= 0.0:
            self.half_width = HALF_WIDTH_BY_OCCUPIED.get(
                self.occupied, self.occupied * SUBLANE_WIDTH / 2.0 - 0.2)

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


def _vehicle_road_state(vehicle):
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


class LeVanLuongSimulator:
    """Le Van Luong simulation with safe (default) or GAMA 2025.6.4 mode."""
    @staticmethod
    @cython.annotation_typing(True)
    def _half_width(vehicle: Vehicle):
        # Physical widths from the model: 0.7 m motorbikes, 1.7 m cars and
        # 2.45 m heavy vehicles (``vehicle_params``), plus the corner reach
        # of a long body on a bending path.
        return vehicle.half_width + vehicle.body_sweep

    @cython.annotation_typing(True)
    def _route_pose(self, vehicle: Vehicle, s):
        """Pose of the vehicle's own sublane path at route coordinate ``s``."""
        road, _ = self._road_state(vehicle)
        if s <= vehicle.approach_end:
            target, progress = vehicle.in_road, s
        elif s <= vehicle.connector_end:
            target, progress = vehicle.conn_road, s - vehicle.approach_end
        else:
            target, progress = vehicle.out_road, s - vehicle.connector_end
        shift = vehicle.current_lateral_shift
        if road == target:
            return self._geometry_roads[target].pose(progress, shift)
        order = (vehicle.in_road, vehicle.conn_road, vehicle.out_road)
        here, there = order.index(road) if road in order else 0, order.index(target)
        # Lateral shifts are road-local: carry the shift across each seam.
        while here < there:
            shift = self._continuous_seam_pose(order[here], order[here + 1], shift)[1]
            here += 1
        if here > there:
            # Backwards across a seam: invert the (affine) forward mapping.
            prev, cur = order[there], order[here]
            a = self._continuous_seam_pose(prev, cur, 0.0)[1]
            b = self._continuous_seam_pose(prev, cur, 1.0)[1]
            shift = (shift - a) / (b - a) if abs(b - a) > 1e-9 else shift
        return self._geometry_roads[target].pose(progress, shift)

    def _walk(self, vehicle, s0, distance, step=0.5):
        """Route coordinate reached from ``s0`` after ``distance`` metres
        (negative: backwards) measured along the vehicle's own sublane path.

        Route coordinates are progress on the roads' reference lines; on the
        outside of a turn a metre of reference progress is up to ~1.35 m of
        path, on the inside ~0.5 m. Body units placed by reference offsets
        drifted apart or into each other on turns.
        """
        if abs(distance) < 1e-9:
            return s0
        direction = 1.0 if distance > 0 else -1.0
        target = abs(distance)
        here = self._route_pose(vehicle, s0)
        travelled, s = 0.0, s0
        for _ in range(int(4.0 * target / step) + 8):
            s_next = s + direction * step
            there = self._route_pose(vehicle, s_next)
            d = math.hypot(there.x - here.x, there.y - here.y)
            if travelled + d >= target:
                return s + direction * step * ((target - travelled) / d if d > 1e-9 else 0.0)
            travelled += d
            here, s = there, s_next
        return s

    def _walk_many(self, vehicle, s0, distances, step=0.5):
        """``[self._walk(vehicle, s0, d) for d in distances]`` for distances
        of one sign, in one pass along the path (identical arithmetic: the
        walks only differ in where they stop)."""
        if not (all(d >= 0 for d in distances) or all(d <= 0 for d in distances)):
            return [self._walk(vehicle, s0, d, step) for d in distances]
        results = [None] * len(distances)
        pending = []
        for k, distance in enumerate(distances):
            if abs(distance) < 1e-9:
                results[k] = s0
            else:
                pending.append((abs(distance), int(4.0 * abs(distance) / step) + 8, k))
        if not pending:
            return results
        direction = 1.0 if distances[pending[0][2]] > 0 else -1.0
        pending.sort()
        here = self._route_pose(vehicle, s0)
        travelled, s = 0.0, s0
        iteration = 0
        index = 0
        while index < len(pending):
            target, limit, k = pending[index]
            if iteration >= limit:
                results[k] = s
                index += 1
                continue
            s_next = s + direction * step
            there = self._route_pose(vehicle, s_next)
            d = math.hypot(there.x - here.x, there.y - here.y)
            while index < len(pending):
                target, limit, k = pending[index]
                if travelled + d >= target:
                    results[k] = s + direction * step * ((target - travelled) / d if d > 1e-9 else 0.0)
                    index += 1
                else:
                    break
            travelled += d
            here, s = there, s_next
            iteration += 1
        return results

    AXLE_STEP = 0.25  # m of front-axle travel per tractrix integration step

    def _axles(self, vehicle):
        """Front axle pose and rear axle point of a two-axle rigid body.

        Kinematic single-track model at low lateral slip: the front axle
        centre follows the vehicle's sublane path; the rear axle stays one
        wheelbase behind it, always moving towards it (a tractrix). The rear
        axle therefore runs inside the front axle's path on a turn
        (off-tracking): R_rear = sqrt(R_front^2 - wheelbase^2), e.g. 9.9 m
        for the SU truck at its 11.6 m minimum radius, whose inner rear
        wheel then runs at 8.7 m (AASHTO: minimum inside radius 8.6 m).
        The rear axle is committed once per tick (``_commit_axles``); this
        integrates from that state to the current position, so guards that
        try several positions within a tick see consistent bodies.
        """
        key = (vehicle.creation_id, vehicle.position, vehicle.current_lateral_shift, self.cycle)
        cached = self._axle_cache.get(key)
        if cached is not None:
            return cached
        s_front = self._walk(vehicle, vehicle.position,
                             vehicle.length / 2.0 - vehicle.front_overhang)
        front = self._route_pose(vehicle, s_front)
        wheelbase = vehicle.wheelbase
        if vehicle.rear_axle is None or not math.isfinite(vehicle.front_axle_s):
            rear_pose = self._route_pose(vehicle, self._walk(vehicle, s_front, -wheelbase))
            rear = (rear_pose.x, rear_pose.y)
        else:
            rear = vehicle.rear_axle
            travel = s_front - vehicle.front_axle_s
            steps = max(1, int(math.ceil(abs(travel) / self.AXLE_STEP)))
            for k in range(1, steps + 1):
                point = (front if k == steps else self._route_pose(
                    vehicle, vehicle.front_axle_s + travel * k / steps))
                dx, dy = point.x - rear[0], point.y - rear[1]
                norm = math.hypot(dx, dy)
                if norm > 1e-9:
                    rear = (point.x - wheelbase * dx / norm, point.y - wheelbase * dy / norm)
        result = (front, rear, s_front)
        if len(self._axle_cache) > 4096:
            self._axle_cache.clear()
        self._axle_cache[key] = result
        return result

    def _commit_axles(self):
        """Store this tick's rear axle of every two-axle body."""
        for vehicle in self.vehicles:
            if vehicle.wheelbase > 0.0:
                _, rear, s_front = self._axles(vehicle)
                vehicle.rear_axle = rear
                vehicle.front_axle_s = s_front
        self._axle_cache.clear()

    def _rigid_pose(self, vehicle):
        """Centre and heading of a two-axle rigid body (one rectangle from
        the rear overhang behind the rear axle to the front overhang ahead
        of the front axle)."""
        front, rear, _ = self._axles(vehicle)
        dx, dy = front.x - rear[0], front.y - rear[1]
        norm = math.hypot(dx, dy) or 1.0
        tx, ty = dx / norm, dy / norm
        rear_overhang = vehicle.length - vehicle.wheelbase - vehicle.front_overhang
        # Body centre: midway between rear end and front end.
        offset = (vehicle.wheelbase + vehicle.front_overhang - rear_overhang) / 2.0
        cx, cy = rear[0] + tx * offset, rear[1] + ty * offset
        return Pose(cx, cy, tx, ty, math.atan2(ty, tx), 0.0)

    def _body_axis_points(self, vehicle):
        """Rear quarter, centre and front quarter of a rigid two-axle body,
        as (x, y, tx, ty)."""
        pose = self._rigid_pose(vehicle)
        quarter = vehicle.length / 4.0
        return tuple((pose.x + pose.tx * k * quarter, pose.y + pose.ty * k * quarter,
                      pose.tx, pose.ty) for k in (-1, 0, 1))

    def _body_polygons(self, vehicle, road=None, inflate=0.0):
        """Footprint of a body (a list with one rectangle).

        Cars and motorbikes: a rigid rectangle tangent to their path at the
        centre, as before. Heavy vehicles: one rigid rectangle placed by its
        two axles (``_rigid_pose``).
        """
        width = 2 * (vehicle.half_width + inflate)
        if vehicle.wheelbase > 0.0:
            return [body_polygon(self._rigid_pose(vehicle), length=vehicle.length, width=width)]
        if road is None:
            road, progress = self._road_state(vehicle)
        else:
            progress = (vehicle.position if road == vehicle.in_road else
                        vehicle.position - vehicle.approach_end if road == vehicle.conn_road
                        else vehicle.position - vehicle.connector_end)
        pose = self._geometry_roads[road].pose(progress, vehicle.current_lateral_shift)
        return [body_polygon(pose, length=vehicle.length, width=width)]

    def _update_body_sweep(self):
        """Sideways reach of a rigid two-axle body beyond its path band.

        On a turn the rear axle runs inside the path (off-tracking) and the
        front overhang swings outside it; at the stop-line jogs the straight
        body cuts the corner. The largest sideways distance between the body
        axis and the path, at the front end, the axles and the rear end, is
        added to the half width used by the lateral rules, so neighbours
        keep clear of the whole swept body. Cars and motorbikes keep the
        model's original band.
        """
        for vehicle in self.vehicles:
            if vehicle.wheelbase <= 0.0:
                continue
            # Everywhere: a lane change yaws the body as well (the front axle
            # moves sideways first, the rear axle follows).
            front, rear, s_front = self._axles(vehicle)
            dx, dy = front.x - rear[0], front.y - rear[1]
            norm = math.hypot(dx, dy) or 1.0
            tx, ty = dx / norm, dy / norm
            rear_overhang = vehicle.length - vehicle.wheelbase - vehicle.front_overhang
            reach = 0.0
            # Distances behind the front axle along the body axis.
            backs = (-vehicle.front_overhang, 0.0, vehicle.wheelbase / 2.0,
                     vehicle.wheelbase, vehicle.wheelbase + rear_overhang)
            # The three walks backwards share one pass along the path.
            walked = self._walk_many(vehicle, s_front, [-b for b in backs[2:]])
            path_s = [self._walk(vehicle, s_front, -backs[0]),
                      self._walk(vehicle, s_front, -backs[1])] + walked
            for back, s_path in zip(backs, path_s):
                ax, ay = front.x - tx * back, front.y - ty * back
                path = self._route_pose(vehicle, s_path)
                side = abs((ax - path.x) * path.ty - (ay - path.y) * path.tx)
                reach = max(reach, side)
            vehicle.body_sweep = min(2.5, reach + 0.1) if reach > 0.05 else 0.0

    def _path_min_radius(self, road, shift, window=8):
        """Tightest radius (m) of a path, measured over ``window`` x 0.5 m."""
        geometry = self._geometry_roads[road]
        poses = [geometry.pose(min(geometry.length, i * 0.5), shift)
                 for i in range(int(geometry.length / 0.5) + 1)]
        best = math.inf
        for i in range(len(poses) - window):
            a, b = poses[i], poses[i + window]
            turn = abs((math.atan2(b.ty, b.tx) - math.atan2(a.ty, a.tx) + math.pi)
                       % (2 * math.pi) - math.pi)
            arc = sum(math.hypot(poses[j + 1].x - poses[j].x, poses[j + 1].y - poses[j].y)
                      for j in range(i, i + window))
            if turn > 1e-6:
                best = min(best, arc / turn)
        return best

    @cython.annotation_typing(True)
    def _interaction_reach(self, vehicle: Vehicle, other: Vehicle, other_position, position):
        """Lateral distance below which ``other`` leads or follows ``vehicle``.

        The corner reach of a long body (``body_sweep``) widens the band for
        bodies ahead of or behind it, so they keep clear of its swinging ends.
        A body already alongside is not turned into a leader with a negative
        gap when the reach grows on a bend; the physical body guards keep the
        two apart. Without long bodies this is the original rule.
        """
        reach: cython.double
        reach = vehicle.half_width + other.half_width + LATERAL_CLEARANCE
        if vehicle.body_sweep or other.body_sweep:
            if (other_position is None or abs(other_position - position) >=
                    (vehicle.length + other.length) / 2):
                reach += vehicle.body_sweep + other.body_sweep
        return reach

    def _length_slack(self):
        """Excess length (m) of the longest vehicle over a passenger car."""
        return max(0.0, self._max_vehicle_length - LIGHT_VEHICLE_LENGTH)

    @cython.annotation_typing(True)
    def _footprint(self, vehicle: Vehicle):
        first, last = vehicle.lane, vehicle.lane + vehicle.occupied
        if vehicle.lane_change_progress < 1.0 and vehicle.lane_change_from >= 0:
            first = min(first, vehicle.lane_change_from)
            last = max(last, vehicle.lane_change_from + vehicle.occupied)
        return range(first, last)

    def _continuous_seam_pose(self, old_road, next_road, old_shift):
        """Return next-road progress and lateral shift at the old center point."""
        # Pure function of the fixed road geometry: memoised.
        key = (old_road, next_road, old_shift)
        cache = self.__dict__.setdefault("_seam_pose_cache", {})
        cached = cache.get(key)
        if cached is not None:
            return cached
        if len(cache) >= 50000:
            cache.clear()
        result = cache[key] = self._continuous_seam_pose_uncached(old_road, next_road, old_shift)
        return result

    def _continuous_seam_pose_uncached(self, old_road, next_road, old_shift):
        old, new = self._geometry_roads[old_road], self._geometry_roads[next_road]
        before = old.pose(old.length, old_shift)
        origin = new.pose(0.0, new.num_lanes * SUBLANE_WIDTH - SUBLANE_WIDTH / 2.0)
        nx, ny = origin.ty, -origin.tx
        dx, dy = before.x - origin.x, before.y - origin.y
        along = dx * origin.tx + dy * origin.ty
        offset = dx * nx + dy * ny
        shift = new.num_lanes * SUBLANE_WIDTH - SUBLANE_WIDTH / 2.0 - offset
        return along, shift

    @cython.annotation_typing(True)
    def _route_target_lane(self, vehicle: Vehicle):
        if vehicle.road != vehicle.in_road:
            return vehicle.lane
        incoming = self.roads[vehicle.in_road]
        minimum, maximum = turn_lane_bounds(
            incoming["num_lanes"], vehicle.occupied, vehicle.maneuver)
        return min(max(vehicle.lane, minimum), maximum)

    @cython.annotation_typing(True)
    def _release_mutual_merge(self, vehicle: Vehicle, buffer_stop):
        """Hold point of a merger blocked by a merger standing beside it.

        Two vehicles that both still have to merge into their route corridor
        can stand side by side at their merge buffers, each sitting in the
        sublanes the other needs: with 3-sublane heavy vehicles the through
        corridor has only two, overlapping, positions, so two heavy vehicles
        abreast never resolve it (seen as a stalled approach at 10 % heavy
        vehicles). The vehicle further ahead (or created first) is let past
        its buffer, one body length ahead of the other, so it can move over
        in front of it; the other then merges behind. The real stop line
        still applies.
        """
        other: Vehicle
        reach_long = (vehicle.length + self._max_vehicle_length) / 2 + 1.0
        release = buffer_stop
        nearby = (self._nearby(vehicle.road, vehicle.position - reach_long,
                               vehicle.position + reach_long)
                  if vehicle.road == vehicle.in_road and getattr(self, "_road_index", None)
                  else self.vehicles)
        for other in nearby:
            if (other is vehicle or other.road != vehicle.road or other.speed >= 0.5 or
                    abs(other.position - vehicle.position) > reach_long or
                    other.lane == self._route_target_lane(other)):
                continue
            if abs(other.position - vehicle.position) >= (vehicle.length + other.length) / 2:
                continue  # not beside each other
            if (vehicle.position, -vehicle.creation_id) < (other.position, -other.creation_id):
                continue  # the other one goes first
            # Is ``other`` in the way of this vehicle's next sublane step?
            step = 1 if self._route_target_lane(vehicle) > vehicle.lane else -1
            target = (vehicle.lane + step + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
            # Same widths as the lane-change sweep check (``_change_lane``):
            # a long body's corner reach blocks the merge there, so it must
            # count here too, or neither vehicle is ever let go first.
            if abs(target - other.current_lateral_shift) >= (
                    self._half_width(vehicle) + self._half_width(other) + LATERAL_CLEARANCE):
                continue
            ahead = other.position + (other.length + vehicle.length) / 2 + max(
                vehicle.gap, other.gap) + 1.0
            release = max(release, ahead)
        return min(release, self._stop_position(vehicle))

    @cached_property
    def _connectors_by_turn(self):
        connectors = {}
        for road in self.roads.values():
            if road["kind"] == "connector":
                connectors.setdefault((road["arm"], road["turn"]), road)
        return connectors

    def _right_target(self, arm):
        return self._connectors_by_turn[arm, "right"]["to_arm"]

    def _mapped_lane(self, vehicle, lane, old_road, next_road):
        old_max = self.roads[old_road]["num_lanes"] - vehicle.occupied
        next_max = self.roads[next_road]["num_lanes"] - vehicle.occupied
        if self.mode == "gama_2025_6_4":
            return min(lane, next_max)
        if vehicle.maneuver == "right":
            # Right-turn connectors are built from the curb (build_layout.py):
            # sublane k of the approach, the connector and the exit coincide.
            if vehicle.wheelbase > 0.0 and next_road == vehicle.conn_road:
                # A heavy vehicle swings out to a sublane it can steer round.
                wide = self._heavy_right_lane.get(next_road)
                return min(next_max, max(lane, wide if wide is not None else next_max))
            return min(lane, next_max)
        # Align all maneuvers to the median edge so discrete sublane IDs match
        # continuous lateral_shift across narrower connectors and exits.
        return min(next_max, max(0, lane + next_max - old_max))

    CURVE_STEP = 0.5       # m of reference progress between profile samples
    CURVE_SHIFT_BIN = 0.35  # m of lateral shift per cached profile

    def _curve_profile(self, road, shift, lat_accel=None):
        """(speed caps, path scales) of ``road`` for a lateral ``shift``.

        Sample ``i`` sits at reference progress ``i * CURVE_STEP``. The speed
        cap is the physical speed the path allows there: sqrt(a_lat * R) of
        the shifted path, lowered so that every tighter point ahead can be
        reached braking at ``curve_decel``. The scale is physical path length
        per metre of reference progress: sublanes on the outside of a turn
        travel further than the reference line, inner ones less.
        """
        lat_accel = self.curve_lat_accel if lat_accel is None else lat_accel
        key = (road, int(round(shift / self.CURVE_SHIFT_BIN)))
        if lat_accel != self.curve_lat_accel:
            key = key + (lat_accel,)
        profile = self._curve_profiles.get(key)
        if profile is not None:
            return profile
        geometry = self._geometry_roads[road]
        length = geometry.length
        count = max(2, int(math.ceil(length / self.CURVE_STEP)) + 1)
        lateral = key[1] * self.CURVE_SHIFT_BIN
        points = []
        for i in range(count):
            pose = geometry.pose(min(length, i * self.CURVE_STEP), lateral)
            points.append((pose.x, pose.y))
        ref_step = [min(length, (i + 1) * self.CURVE_STEP) - min(length, i * self.CURVE_STEP)
                    for i in range(count - 1)]
        seg = [math.dist(a, b) for a, b in zip(points, points[1:])]
        heading = [math.atan2(b[1] - a[1], b[0] - a[0]) if d > 1e-9 else 0.0
                   for (a, b), d in zip(zip(points, points[1:]), seg)]
        # +-4 m: a polyline concentrates its turning at vertices, and a driver
        # smooths short jogs of the lane alignment over about a car length.
        window = 8
        if self.roads[road].get("turn") == "through":
            # Through connectors are straight apart from alignment jogs of
            # about a metre near the stop lines; drivers straighten those.
            window = 20
        caps, scales = [], []
        road_cap = 40.0 / 3.6
        for i in range(count):
            lo, hi = max(0, i - window), min(count - 2, i + window - 1)
            arc = sum(seg[lo:hi + 1])
            ref = sum(ref_step[lo:hi + 1])
            turn = abs((heading[hi] - heading[lo] + math.pi) % (2 * math.pi) - math.pi)
            # ``turn`` is accumulated between the first and last segment
            # centres, i.e. over the arc minus half of each end segment.
            span = arc - 0.5 * (seg[lo] + seg[hi])
            if lat_accel > 0 and turn > 1e-4 and span > 1e-6:
                caps.append(min(road_cap, math.sqrt(lat_accel * span / turn)))
            else:
                caps.append(road_cap)
            scales.append(min(1.6, max(0.5, arc / ref)) if ref > 1e-9 else 1.0)
        for i in range(count - 2, -1, -1):
            caps[i] = min(caps[i], math.sqrt(
                caps[i + 1] ** 2 + 2.0 * self.curve_decel * seg[i]))
        profile = (tuple(caps), tuple(scales))
        self._curve_profiles[key] = profile
        return profile

    @cython.annotation_typing(True)
    def _path_scale(self, vehicle: Vehicle):
        """Physical metres travelled per metre of route progress."""
        if (self.mode != "safe" or
                not vehicle.approach_end <= vehicle.position < vehicle.connector_end):
            return 1.0
        scales = self._curve_profile(vehicle.conn_road, vehicle.current_lateral_shift)[1]
        index = int((vehicle.position - vehicle.approach_end) / self.CURVE_STEP)
        return scales[min(len(scales) - 1, max(0, index))]

    @cython.annotation_typing(True)
    def _curve_speed_cap(self, vehicle: Vehicle):
        """Speed the vehicle's path geometry allows here (m/s, physical)."""
        distance: cython.double
        if self.curve_lat_accel <= 0:
            return math.inf
        if vehicle.position >= vehicle.connector_end:
            return math.inf
        # Per-type lateral acceleration (heavy vehicles: passenger comfort).
        lat_accel = (min(vehicle.lat_accel, self.curve_lat_accel)
                     if vehicle.lat_accel > 0 else None)
        if vehicle.position >= vehicle.approach_end:
            caps = self._curve_profile(vehicle.conn_road, vehicle.current_lateral_shift,
                                       lat_accel)[0]
            index = int((vehicle.position - vehicle.approach_end) / self.CURVE_STEP)
            return caps[min(len(caps) - 1, max(0, index))]
        distance = vehicle.approach_end - vehicle.position
        if distance > 60.0:
            return math.inf
        key = (vehicle.conn_road, int(round(vehicle.current_lateral_shift / self.CURVE_SHIFT_BIN)))
        if lat_accel is not None:
            key = key + (lat_accel, self._mapped_lane(vehicle, vehicle.lane, vehicle.in_road,
                                                      vehicle.conn_road))
        entry = self._curve_entry.get(key)
        if entry is None:
            seam_shift = self._continuous_seam_pose(
                vehicle.in_road, vehicle.conn_road, key[1] * self.CURVE_SHIFT_BIN)[1]
            lanes = self.roads[vehicle.conn_road]["num_lanes"]
            seam_shift = min(max(seam_shift, 0.0), (lanes - 1) * SUBLANE_WIDTH)
            if vehicle.wheelbase > 0.0:
                # It brakes for the sublane it will actually drive.
                lane = self._mapped_lane(vehicle, vehicle.lane, vehicle.in_road, vehicle.conn_road)
                seam_shift = (lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
            entry = self._curve_profile(vehicle.conn_road, seam_shift, lat_accel)[0][0]
            self._curve_entry[key] = entry
        return math.sqrt(entry * entry + 2.0 * self.curve_decel * distance)

    @staticmethod
    @cython.annotation_typing(True)
    def _stop_position(vehicle: Vehicle):
        return (vehicle.approach_end - vehicle.length / 2 - STOP_MARGIN
                - vehicle.queue_stop_offset)

    def __init__(self, seed=1, flow_per_approach=2400.0,
                 motorbike_ratio=.85, p_left=.2, p_right=.2,
                 green_lvl=50.0, green_hdt=40.0, all_red=2.0,
                 roads_path=ROOT / "includes" / "roads.geojson",
                 demand_file=None, train_mode=False, mode="safe", yellow=4.0,
                 vehicle_params=None, startup_params=None,
                 junction_model="negotiation", negotiation_params=None,
                 min_green=10.0, jam_timeout=120.0,
                 curve_lat_accel=3.0, curve_decel=2.0, max_decel=7.5,
                 heavy_ratio=DEFAULT_HEAVY_RATIO):
        if mode not in ("safe", "gama_2025_6_4"):
            raise ValueError("mode must be 'safe' or 'gama_2025_6_4'")
        self.mode = mode
        self.seed = int(seed)
        numeric_parameters = (flow_per_approach, motorbike_ratio, p_left,
                              p_right, green_lvl, green_hdt, all_red, heavy_ratio)
        if (not all(math.isfinite(value) for value in numeric_parameters) or
                flow_per_approach < 0 or not 0 <= motorbike_ratio <= 1 or
                not 0 <= heavy_ratio <= 1 or motorbike_ratio + heavy_ratio > 1 + 1e-12 or
                min(p_left, p_right, green_lvl, green_hdt, all_red) < 0 or
                p_left + p_right > 1 or flow_per_approach / 3600 * DT > 1):
            raise ValueError("Invalid flow, vehicle mix, turn split, or signal timing")
        self.rng = random.Random(seed)
        # Vehicle types first: the lane-level movements below need the number
        # of sublanes each type occupies.
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
        self._occupied = {kind: occupied_sublanes(self.vehicle_params[kind]["width"])
                          for kind in VEHICLE_KINDS}
        # Lane movement ids and several behaviour rules identify the type by
        # its sublane count, so the three types must keep distinct counts.
        if not (self._occupied["motorbike"] == 1 < self._occupied["car"]
                < self._occupied["heavy"]):
            raise ValueError(
                "vehicle widths must give 1 sublane for motorbikes and more "
                "sublanes for cars than for motorbikes, for heavy vehicles than for cars")
        self._half_width_by_occupied = {
            self._occupied[kind]: self.vehicle_params[kind]["width"] / 2.0
            for kind in VEHICLE_KINDS}
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
            for occupied in sorted(self._half_width_by_occupied):
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
        self._axle_cache = {}
        self.heavy_turns_rerouted = 0
        # Right-turn connector sublane a heavy vehicle drives (front axle):
        # the curb-most one whose tightest radius allows its minimum
        # turning radius; None when no sublane does (the turn is then
        # rerouted, see ``_create_vehicle``).
        self._heavy_right_lane = {}
        heavy = self.vehicle_params["heavy"]
        heavy_occupied = self._occupied["heavy"]
        for road_id, road in self.roads.items():
            if road.get("kind") != "connector" or road.get("turn") != "right":
                continue
            lane = None
            for candidate in range(max(0, road["num_lanes"] - heavy_occupied + 1)):
                shift = (candidate + (heavy_occupied - 1) / 2.0) * SUBLANE_WIDTH
                if self._path_min_radius(road_id, shift) >= heavy["min_turn_radius"] - 0.05:
                    lane = candidate
                    break
            self._heavy_right_lane[road_id] = lane
        self._lane_conflict_pair_cache = {}
        self._conflict_engine = ConflictEngine(
            self._conflict_map, left_wait_priority_after=10.0)
        self._junction_grants = set()
        self._road_index = None
        self._exited_this_tick = []
        # Upper bounds used to size index windows (updated on creation).
        self._max_vehicle_length = LIGHT_VEHICLE_LENGTH
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
        self.heavy_ratio = heavy_ratio
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
        # Train mode: a phase must have shown green for ``min_green`` seconds
        # before the policy may leave it (see ``action_mask``).
        min_green = float(min_green)
        jam_timeout = float(jam_timeout)
        if not math.isfinite(min_green) or min_green < 0 or math.isnan(jam_timeout) or jam_timeout < 0:
            raise ValueError("min_green and jam_timeout must be non-negative")
        self.min_green = min_green
        self._min_green_ticks = int(round(min_green / DT))
        # Safety net (cf. SUMO time-to-teleport): a vehicle standing inside
        # the junction for longer than this is removed and counted. 0 or inf
        # disables it. It must stay at zero in a healthy run.
        self.jam_timeout = jam_timeout
        self.jam_removed = 0
        # Kinematic limits (safe mode). ``curve_lat_accel`` caps the lateral
        # acceleration on curved connector paths (v <= sqrt(a_lat * R)) with
        # anticipatory braking at ``curve_decel`` before the curve; 0 turns
        # the limit off. ``max_decel`` is the emergency braking bound of the
        # car-following law; position guards that need more than this are
        # counted in ``hard_brake_events`` instead of being hidden.
        for name, value in (("curve_lat_accel", curve_lat_accel),
                            ("curve_decel", curve_decel), ("max_decel", max_decel)):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a non-negative number")
        if float(curve_decel) <= 0 or float(max_decel) <= 0:
            raise ValueError("curve_decel and max_decel must be positive")
        self.curve_lat_accel = float(curve_lat_accel) if self.mode == "safe" else 0.0
        self.curve_decel = float(curve_decel)
        self.max_decel = float(max_decel)
        self._curve_profiles = {}
        self._curve_entry = {}
        self.hard_brake_events = 0
        self.masked_actions = 0
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
                                not 0 <= row[2] < len(VEHICLE_KINDS) or
                                not math.isfinite(row[3]) or row[3] <= 0 or
                                not 0 <= row[4] <= 2):
                            raise ValueError(f"Invalid demand row: {raw}")
                        occupied = self._occupied[VEHICLE_KINDS[row[2]]]
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

    def action_mask(self):
        """Legal actions (LVL, -, HDT, -) at the current decision tick.

        Keeping the running phase is always legal. Leaving it requires
        ``min_green`` seconds of green; during yellow/all-red the pending
        phase is the only legal choice.
        """
        if self.phase in ("LVL", "HDT"):
            current = self.phase
            may_switch = self.green_ticks >= self._min_green_ticks
        else:
            current = self.pending_phase
            may_switch = False
        return (current == "LVL" or may_switch, False,
                current == "HDT" or may_switch, False)

    def set_action(self, action):
        if not self.train_mode or not self.dqn_waiting_action or action not in (0, 2):
            raise ValueError("Action requires a Le Van Luong decision tick and phase 0 or 2")
        if not self.action_mask()[action]:
            # An illegal switch is a hold, never an error: the environment
            # masks it, but baselines and old checkpoints may still ask.
            self.masked_actions += 1
            action = 0 if self.action_mask()[0] else 2
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
                _, origin, kind, speed, turn, lane = self._rows[self._cursor]
                self._create_vehicle(ARMS[origin], ("left", "right", "through")[turn],
                                     VEHICLE_KINDS[kind], lane, speed, self._cursor)
                self._cursor += 1
            return
        for arm in ARMS:
            if self.rng.random() >= self.flow_per_approach / 3600 * DT:
                continue
            draw = self.rng.random()
            turn = "left" if draw < self.p_left else (
                "right" if draw < self.p_left + self.p_right else "through")
            incoming = self.roads[f"{arm}_in"]
            kind = kind_from_draw(self.rng.random(), self.motorbike_ratio,
                                  self.heavy_ratio)
            occupied = self._occupied[kind]
            minimum, maximum = (0, incoming["num_lanes"] - occupied) if turn == "through" else turn_lane_bounds(incoming["num_lanes"], occupied, turn)
            lane = self.rng.randrange(minimum, maximum + 1)
            params = self.vehicle_params[kind]
            speed_kmh = max(1.0, self.rng.gauss(params["speed_mean"], params["speed_sd"]))
            self._create_vehicle(arm, turn, kind, lane, speed_kmh)

    def _create_vehicle(self, arm, turn, kind, lane, speed_kmh, demand_row_id=-1):
        """Add a vehicle. ``kind`` is a type name ("motorbike", "car",
        "heavy"), a demand code (0, 1, 2) or the legacy car flag (bool)."""
        kind = vehicle_kind(kind)
        is_car = kind != "motorbike"
        if (kind == "heavy" and turn == "right" and self.mode == "safe" and
                self._heavy_right_lane.get(f"{arm}_{self._right_target(arm)}") is None):
            # No sublane of this right turn is wide enough for the heavy
            # vehicle's minimum turning radius: it goes straight on instead.
            turn = "through"
            self.heavy_turns_rerouted += 1
        connector = self._connectors_by_turn[arm, turn]
        target = connector["to_arm"]
        incoming = self.roads[f"{arm}_in"]
        outgoing = self.roads[f"{target}_out"]
        kind_params = self.vehicle_params[kind]
        occupied = self._occupied[kind]
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
        length = kind_params["length"]
        position = 0.0
        new_half_width = kind_params["width"] / 2.0
        for other in self.vehicles if self.mode == "safe" else ():
            if other.road != f"{arm}_in":
                continue
            if abs(initial_shift - other.current_lateral_shift) >= (
                    new_half_width + self._half_width(other) + LATERAL_CLEARANCE):
                continue
            position = min(position, other.position - (length + other.length) / 2
                           - max(1.0, other.gap))
        # Vehicles arrive from upstream at speed, not from rest at the model
        # boundary. Enter at the desired speed, lowered to what the headway
        # to the vehicle ahead in the same lateral band allows (IDM steady
        # state and the ability to stop behind it at comfortable braking).
        entry_speed = 0.0
        if self.mode == "safe":
            entry_speed = min(speed_kmh / 3.6, 60.0 / 3.6)
            for other in self.vehicles:
                if other.road != f"{arm}_in" or other.position < position:
                    continue
                if abs(initial_shift - other.current_lateral_shift) >= (
                        new_half_width + self._half_width(other) + LATERAL_CLEARANCE):
                    continue
                room = (other.position - position - (length + other.length) / 2
                        - kind_params["gap"])
                if room <= 0.0:
                    entry_speed = 0.0
                    break
                entry_speed = min(
                    entry_speed, room / kind_params["headway"],
                    math.sqrt(other.speed * other.speed +
                              2.0 * kind_params["decel"] * room))
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
            speed=entry_speed,
            demand_row_id=demand_row_id,
            current_lateral_shift=initial_shift,
            start_lateral_shift=initial_shift,
            target_lateral_shift=initial_shift,
            lane_change_progress=1.0,
            lateral_duration=LATERAL_DURATION[kind],
            queue_stop_offset=queue_stop_offset,
            kind=kind, half_width=new_half_width,
            wheelbase=(kind_params["length"] - kind_params["front_overhang"] -
                       kind_params["rear_overhang"]) if "front_overhang" in kind_params else 0.0,
            front_overhang=kind_params.get("front_overhang", 0.0),
            lat_accel=kind_params.get("lat_accel", 0.0),
            time_since_lane_change=0.8 if initial_merge_ready else 0.0,
            gama_entered=self.mode != "gama_2025_6_4"))
        self.created += 1

    @cython.annotation_typing(True)
    def _destination_has_room(self, vehicle: Vehicle):
        """Require the complete vehicle plus a buffer to fit beyond the connector."""
        other: Vehicle
        target = vehicle.to_arm
        target_road = f"{target}_out"
        out_length = self.roads[target_road]["length"]
        target_shift = (self._mapped_lane(vehicle, vehicle.lane,
                                          vehicle.in_road,
                                          vehicle.conn_road) +
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
        base = vehicle.conn_road
        lane = (vehicle.lane if on_connector else self._mapped_lane(
            vehicle, vehicle.lane, vehicle.in_road, base))
        return f"{base}@{lane}:{vehicle.occupied}"

    @cython.annotation_typing(True)
    def _leader_advance(self, leader: Vehicle | None, road, moved_ids):
        """Distance the leader is still expected to cover this tick.

        A leader processed earlier in this sequential pass has already moved
        and its lane-map entry holds the new position, so adding its advance
        again would let the follower encroach and be pushed back afterwards.
        A leader without permission to cross the stop line will not pass its
        stop point: counting a full step of it let the follower close in on
        a leader that then stopped short, and the road guard pushed the
        follower back (a few cm, 16/11/2026).
        """
        if (leader is None or id(leader) in moved_ids or leader.road != road or
                leader.speed <= 0.5):
            return 0.0
        advance = max(0.0, leader.speed * DT - 0.5 * leader.decel * DT * DT)
        if leader.leading_heavy:
            # The leader follows a truck or bus, which can stop short (it is
            # held by a junction stop, or yaws through a lane change): count
            # no more than the leader's own room.
            room = leader.leading_distance
            road, position = leader.leading_from
            if road == leader.road:
                # The gap was measured before the leader's own last move.
                room -= leader.position - position
            advance = min(advance, max(0.0, room - max(1.0, leader.gap)))
        if (self.mode == "safe" and road == leader.in_road and
                not leader.junction_entry_token and
                leader.creation_id not in self._junction_grants):
            stop = self._stop_position(leader)
            room = stop - leader.position
            if -1e-8 <= room <= leader.speed * DT + leader.speed ** 2 / (2.0 * leader.decel):
                # It is about to hold at its stop point and may brake up to
                # ``max_decel`` to do so (a signal change revokes grants).
                hard_stop = (leader.speed ** 2 / (2.0 * self.max_decel)
                             if leader.speed <= self.max_decel * DT else
                             leader.speed * DT - 0.5 * self.max_decel * DT * DT)
                advance = min(advance, max(0.0, room), hard_stop)
        return advance

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

    @cython.annotation_typing(True)
    def _can_enter_lane(self, vehicle: Vehicle):
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

    @cython.annotation_typing(True)
    def _green_for(self, vehicle: Vehicle):
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
        if vehicle.kind != "motorbike":
            utility += c[vehicle.kind]
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
        cap_now = self._curve_speed_cap(vehicle)
        if math.isfinite(cap_now):
            # A turning vehicle slows for its curve: its mean speed over the
            # remaining approach lies between the cap here and at the line.
            cap_line = math.sqrt(max(0.0, cap_now * cap_now -
                                     2.0 * self.curve_decel * max(0.0, x)))
            speed_limit = min(speed_limit, 0.5 * (cap_now + cap_line))
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
            if not (vehicle.junction_entry_token and vehicle.road == vehicle.in_road
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
            base_movement_id = vehicle.conn_road
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
            if road != vehicle.in_road:
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
        lane_conflicts.default_vehicle_length = LIGHT_VEHICLE_LENGTH
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
                        half_a = self._half_width_by_occupied[occ_a]
                        half_b = self._half_width_by_occupied[occ_b]
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

    @cython.annotation_typing(True)
    def _move(self):
        old_progress: cython.double
        new_progress: cython.double
        new_distance: cython.double
        new_shift: cython.double
        old_shift: cython.double
        road_progress: cython.double
        seam_progress: cython.double
        remaining: cython.double
        crossing_speed: cython.double
        new_scale: cython.double
        half_length: cython.double
        lateral_gap: cython.double
        separation: cython.double
        needed: cython.double
        closing: cython.double
        other_shift: cython.double
        ego_cur: cython.double
        ego_next: cython.double
        lo: cython.double
        hi: cython.double
        remaining_change: cython.double
        vehicle: Vehicle
        ahead: Vehicle
        other: Vehicle
        item: Vehicle
        distance: cython.double
        initial_speed: cython.double
        scale: cython.double
        unconstrained_distance: cython.double
        to_boundary: cython.double
        seam_gap: cython.double
        entry_horizon: cython.double
        predicted_speed: cython.double
        future_time: cython.double
        acceleration: cython.double
        can_cross: cython.bint
        is_in_road: cython.bint
        route_invalid: cython.bint
        committed: cython.bint
        safe_committed: cython.bint

        if self.mode == "safe":
            self._update_body_sweep()
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
        tick_time = self.cycle * DT
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
                scheduler = from_startup[arm]
                if green_for_arm == scheduler.green:
                    continue
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
                scheduler.update(tick_time, green_for_arm, snapshot)
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
        # Only a vehicle whose nose has crossed its exit seam can occupy
        # downstream space. Add new arrivals as each vehicle finishes moving.
        dest_groups = {arm: [] for arm in ARMS}
        if self.mode == "safe":
            for item in self.vehicles:
                if item.position - item.connector_end + item.length / 2 > 0.0:
                    dest_groups[item.to_arm].append(item)
            if self._negotiation is not None:
                self._dest_groups = dest_groups
                self._negotiation.step(DT)
            else:
                self._reserve_junction()
        self._dest_groups = dest_groups
        for vehicle in self.vehicles:
            pending_entry = self.mode == "gama_2025_6_4" and not vehicle.gama_entered
            if pending_entry:
                road, progress = vehicle.in_road, 0.0
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
            entry = (progress, vehicle.creation_id, vehicle)
            footprint = self._footprint(vehicle)
            lanes = lane_maps[road]
            for sub in footprint:
                lanes[sub].append(entry)
            if self.mode == "safe":
                index = self._road_index[road]
                index.entries.append(entry)
                index.counts[progress, vehicle.creation_id] = len(footprint)
        # No queries occur during loading. Sort once before the first lookup;
        # subsequent moves use the normal live insert/pop path.
        for lanes in lane_maps.values():
            for lane in lanes:
                lane.sort()
        for index in self._road_index.values():
            index.entries.sort()
        survivors = []
        moved_ids = set()
        seam_index = (_SeamFollowIndex(self.vehicles)
                      if self.mode == "safe" and self.junction_model == "negotiation" else None)
        for order, vehicle in enumerate(self.vehicles):
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
                if vehicle.merge_wait >= MERGE_PATIENCE:
                    stop_position = self._release_mutual_merge(vehicle, stop_position)
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
                    # A vehicle created this tick is not in the scheduler's
                    # snapshot yet; it arrives at speed and is not a member
                    # of the standing queue the gate releases.
                    vehicle.created_cycle < self.cycle and
                    (from_startup[vehicle.arm].release_time(vehicle.creation_id) or 0.0) > tick_time):
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

            self._neighbors_reuse = None
            lane_decision = self._choose_lane(vehicle, lane_maps)
            move_lane = vehicle.lane
            reuse = self._neighbors_reuse
            body_shift = None
            if self.mode == "safe":
                lane_centre = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                if abs(vehicle.current_lateral_shift - lane_centre) > 1e-6:
                    body_shift = vehicle.current_lateral_shift
            if body_shift is not None:
                # Between sublanes, the gap at the body's actual position governs.
                leader, gap, _, _ = self._neighbors(
                    vehicle, vehicle.lane, lane_maps, shift=body_shift)
            elif reuse is not None and reuse[0] is vehicle and reuse[1] == vehicle.lane:
                leader, gap, _, _ = reuse[2]
            else:
                leader, gap, _, _ = self._neighbors(vehicle, vehicle.lane, lane_maps)
            if safe_committed and not green and (leader is None or leader.road != old_road):
                # A vehicle whose front crossed the stop line on green must
                # clear the approach during all-red. Connector conflicts are
                # handled separately; they must not trap it on the stop line.
                leader, gap = None, math.inf
            seam_cell = None
            if seam_index is not None and (is_in_road or old_road == vehicle.conn_road):
                # Keep following a same-approach body across the stop-line
                # seam, even when it takes another connector. Those paths
                # share the first metres of the junction and only diverge
                # after their bodies have cleared one another.
                own_pose = None
                seam_cell = math.floor(vehicle.position / SEAM_FOLLOW_DISTANCE)
                for candidate in seam_index.by_arm[vehicle.arm][seam_cell]:
                    ahead = self.vehicles[candidate]
                    if (ahead.position <= vehicle.position or
                            ahead.position - vehicle.position > SEAM_FOLLOW_DISTANCE):
                        continue
                    seam_gap = (ahead.position - vehicle.position -
                                (ahead.length + vehicle.length) / 2.0)
                    if is_in_road and ahead.road == ahead.conn_road and ahead.wheelbase > 0.0:
                        seam_gap -= ahead.body_sweep
                    # A candidate can replace the leader only at a strictly smaller gap.
                    # Test that distance before computing poses and divergence zones.
                    if seam_gap >= gap and (
                            ahead.conn_road == vehicle.conn_road or
                            (vehicle.occupied == ahead.occupied == 1 and
                             not vehicle.evading and not ahead.evading)):
                        continue
                    if ahead.road == old_road:
                        lateral = abs(ahead.current_lateral_shift -
                                      vehicle.current_lateral_shift)
                    else:
                        # Lateral shifts are road-local (each connector has
                        # its own origin), so across the seam the sideways
                        # separation is measured in the plane. Comparing raw
                        # shifts made a vehicle beside this one a "leader"
                        # with a negative gap: a stop from speed in one tick.
                        if own_pose is None:
                            own_pose = self._geometry_roads[old_road].pose(
                                old_progress, vehicle.current_lateral_shift)
                        ahead_road, ahead_progress = self._road_state(ahead)
                        if ahead.wheelbase > 0.0:
                            # Long rigid body: its rear may still be right
                            # ahead while its centre has turned away.
                            ahead_poses = [Pose(x, y, tx, ty, 0.0, 0.0) for x, y, tx, ty
                                           in self._body_axis_points(ahead)]
                        else:
                            ahead_poses = [self._geometry_roads[ahead_road].pose(
                                ahead_progress, ahead.current_lateral_shift)]
                        lateral = min(abs((ahead_pose.y - own_pose.y) * own_pose.tx -
                                          (ahead_pose.x - own_pose.x) * own_pose.ty)
                                      for ahead_pose in ahead_poses)
                    if lateral >= self._interaction_reach(
                            vehicle, ahead, ahead.position, vehicle.position):
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
                    if ((ahead.wheelbase > 0.0 or vehicle.wheelbase > 0.0) and
                            ahead.position <= vehicle.position + vehicle.length / 2.0):
                        # Beside a long body, not behind it: the other
                        # centre is still alongside (see the seam targets in
                        # junction_negotiation).
                        continue
                    if seam_gap < gap:
                        leader, gap = ahead, seam_gap
            vehicle.leading_distance = gap
            vehicle.leading_heavy = leader is not None and leader.wheelbase > 0.0
            vehicle.leading_from = (vehicle.road, vehicle.position)
            vehicle.leading_speed = leader.speed if leader else 0.0
            # Route progress is measured on the road's reference line; on a
            # curved connector a shifted sublane covers ``scale`` physical
            # metres per metre of progress. Speeds stay physical.
            scale = self._path_scale(vehicle)
            acceleration = (lane_decision.acceleration if self.mode == "gama_2025_6_4"
                            else self._idm(vehicle, gap * scale,
                                           leader.speed if leader else 0.0))
            junction_stop = (self._junction_stop.get(vehicle.creation_id)
                             if self._negotiation is not None else None)
            if junction_stop is not None:
                # Yield to a junction foe: approach the stop point before the
                # conflict zone with the ordinary IDM (SUMO adaptToJunctionLeader).
                junction_stop, junction_obstacle_speed = junction_stop
                junction_room = junction_stop - (vehicle.position + vehicle.length / 2.0)
                acceleration = min(acceleration, self._idm(
                    vehicle, max(0.0, junction_room) * scale + vehicle.gap,
                    junction_obstacle_speed))
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
                        # Comfortable braking where it suffices, but never less
                        # than the stop needs: a floor below the kinematic
                        # requirement left residual speed at the hold point,
                        # which the position guard then removed in one tick.
                        needed = vehicle.speed * vehicle.speed / (2.0 * stop_gap)
                        floor = min(self.max_decel,
                                    max(vehicle.decel * 1.4, 4.2, 1.1 * needed))
                        stop_accel = max(-floor, stop_accel)
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
                spawn_entry = (old_progress, vehicle.creation_id, vehicle)
                for sub in self._footprint(vehicle):
                    bisect.insort(lane_maps[old_road][sub], spawn_entry)
            if predicted_speed < 0:
                distance = -0.5 * initial_speed * initial_speed / acceleration
                vehicle.speed = 0.0
            else:
                distance = initial_speed * DT + 0.5 * acceleration * DT * DT
                vehicle.speed = predicted_speed
            distance /= scale
            unconstrained_distance = distance
            # The safety resolver is a final guard, not a way to brake by
            # moving a vehicle backwards. Preserve the required link clearance
            # during forward motion, including connector-to-exit approaches.
            if self.mode == "safe" and math.isfinite(gap):
                room = gap + self._leader_advance(leader, old_road, moved_ids)
                if leader is not None and (old_road.endswith(("_in", "_out")) or
                                           leader.road.endswith("_out")):
                    room -= max(1.0, vehicle.gap, leader.gap)
                distance = min(distance, max(0.0, room))
            if junction_stop is not None:
                # Hard guard: the front never passes the junction stop point.
                junction_room = max(0.0, junction_stop - (vehicle.position + vehicle.length / 2.0)
                                    + junction_obstacle_speed * DT)
                if distance > junction_room + 1e-9:
                    distance = junction_room
                    vehicle.speed = max(0.0, min(
                        vehicle.speed, 2.0 * junction_room * scale / DT - initial_speed))
                    self._negotiation.hard_caps += 1
            if (route_invalid and vehicle.position <= stop_position <
                    vehicle.position + distance):
                distance = max(0.0, stop_position - vehicle.position)
                vehicle.speed = 0.0
            boundary = (vehicle.approach_end if is_in_road else
                        vehicle.connector_end if old_road not in
                        (vehicle.out_road,) else vehicle.route_end)
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
                remaining = min(DT, max(0.0, DT - to_boundary * scale / predicted_speed))
                pre_transition_speed = initial_speed
                pre_transition_lane = vehicle.lane
                old_entry = (old_progress, vehicle.creation_id, vehicle)
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
                    vehicle.road = vehicle.conn_road
                else:
                    vehicle.road = vehicle.out_road
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
                    lane_centre = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                    swing = (vehicle.wheelbase > 0.0 and next_road == vehicle.conn_road and
                             vehicle.maneuver == "right" and abs(new_shift - lane_centre) > 0.3)
                    if new_shift < min_shift - 0.05 or new_shift > max_shift + 0.05 or swing:
                        vehicle.target_lateral_shift = lane_centre
                        vehicle.lane_change_progress = 0.0
                        vehicle.lateral_duration = max(0.8, abs(vehicle.target_lateral_shift - new_shift) / 1.8)
                    else:
                        vehicle.target_lateral_shift = new_shift
                        vehicle.lane_change_progress = 1.0
                if next_road not in lane_maps:
                    lane_maps[next_road] = self._new_road_lanes(next_road)
                self._neighbor_road_state[id(vehicle)] = (next_road, seam_progress)
                self._free_idm_cache.pop(id(vehicle), None)
                new_entry = (seam_progress, vehicle.creation_id, vehicle)
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
                        new_entry = (0.0, vehicle.creation_id, vehicle)
                        for sub in self._footprint(vehicle):
                            entries = lane_maps[next_road][sub]
                            index = bisect.bisect_left(entries, new_entry)
                            if index < len(entries) and entries[index] == new_entry:
                                entries.pop(index)
                        self._neighbor_road_state[id(vehicle)] = (old_road, old_progress)
                        restored_entry = (old_progress, vehicle.creation_id, vehicle)
                        for sub in self._footprint(vehicle):
                            bisect.insort(lane_maps[old_road][sub], restored_entry)
                        survivors.append(vehicle)
                        continue
                    else:
                        new_distance = transition.new_road_distance
                        vehicle.speed = transition.stored_speed
                else:
                    # Speed at the seam: the old road's acceleration acts only
                    # until the boundary is reached. Using the full-step
                    # ``predicted_speed`` here applied it twice (up to 2x
                    # a_max across every road seam).
                    crossing_speed = max(0.0, initial_speed +
                                         acceleration * (DT - remaining))
                    if crossing_speed + new_acceleration * remaining < 0:
                        new_distance = -0.5 * crossing_speed * crossing_speed / new_acceleration
                        vehicle.speed = 0.0
                    else:
                        new_distance = (crossing_speed * remaining +
                                        0.5 * new_acceleration * remaining * remaining)
                        vehicle.speed = max(0.0, crossing_speed +
                                            new_acceleration * remaining)
                new_scale = self._path_scale(vehicle)
                new_distance /= new_scale
                unconstrained_new_distance = new_distance
                if self.mode == "safe" and math.isfinite(new_gap):
                    new_distance = min(new_distance, max(0.0, new_gap))
                if (self.mode == "safe" and remaining > 0 and
                        new_distance < unconstrained_new_distance - 1e-9):
                    vehicle.speed = min(vehicle.speed, max(
                        0.0, 2.0 * new_distance * new_scale / remaining - crossing_speed))
                vehicle.position += max(0.0, new_distance)
                curr_entry = (seam_progress, vehicle.creation_id, vehicle)
                for sub in self._footprint(vehicle):
                    entries = lane_maps[next_road][sub]
                    index = bisect.bisect_left(entries, curr_entry)
                    if index < len(entries) and entries[index] == curr_entry:
                        entries.pop(index)
                final_progress = seam_progress + new_distance
                final_entry = (final_progress, vehicle.creation_id, vehicle)
                for sub in self._footprint(vehicle):
                    bisect.insort(lane_maps[next_road][sub], final_entry)
                self._neighbor_road_state[id(vehicle)] = (next_road, final_progress)
                new_road, new_entry = next_road, final_entry
            else:
                if distance < unconstrained_distance - 1e-9:
                    vehicle.speed = min(vehicle.speed, max(
                        0.0, 2.0 * distance * scale / DT - initial_speed))
                hold_from = vehicle.position
                vehicle.position += max(0.0, distance)
                if is_in_road and not can_cross and vehicle.position > stop_position:
                    # ``stop_position`` may be a merge buffer upstream of the
                    # line; a vehicle already beyond it holds where it is.
                    vehicle.position = max(hold_from, stop_position)
                    vehicle.speed = 0.0
                self._free_idm_cache.pop(id(vehicle), None)
                new_road, new_progress = self._road_state(vehicle)
                vehicle.road = new_road
                self._neighbor_road_state[id(vehicle)] = (new_road, new_progress)
                old_entry = (old_progress, vehicle.creation_id, vehicle)
                new_entry = (new_progress, vehicle.creation_id, vehicle)
                if new_road == old_road and self.mode == "safe":
                    # Lane changes have already updated the complete footprint.
                    # Move the road index once for all occupied sublanes.
                    self._road_index[old_road].move(
                        lane_maps[old_road], self._footprint(vehicle), old_entry, new_entry)
                else:
                    for sub in self._footprint(vehicle):
                        entries = lane_maps[old_road][sub]
                        index = bisect.bisect_left(entries, old_entry)
                        if index < len(entries) and entries[index] == old_entry:
                            entries.pop(index)
                    if new_road != old_road:
                        max_lane = max(0, self.roads[new_road]["num_lanes"] - vehicle.occupied)
                        if vehicle.lane > max_lane:
                            vehicle.lane = max_lane
                            new_shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                            vehicle.start_lateral_shift = vehicle.target_lateral_shift = new_shift
                            vehicle.current_lateral_shift = new_shift
                            vehicle.lane_change_progress = 1.0
                            vehicle.lane_change_from = -1
                    if new_road not in lane_maps:
                        lane_maps[new_road] = self._new_road_lanes(new_road)
                    for sub in self._footprint(vehicle):
                        bisect.insort(lane_maps[new_road][sub], new_entry)
            if seam_cell is not None:
                seam_index.update(order, vehicle, seam_cell)
            if self.mode == "safe":
                half_length = vehicle.length / 2
                if (vehicle.position - vehicle.connector_end + half_length > 0.0 and
                        starting_positions[vehicle.creation_id] - vehicle.connector_end
                        + half_length <= 0.0):
                    dest_groups[vehicle.to_arm].append(vehicle)
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
                    if (new_road == vehicle.in_road and
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
                    own_lo = min(vehicle.current_lateral_shift, next_shift)
                    own_hi = max(vehicle.current_lateral_shift, next_shift)
                    own_progress = road_progress
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
                    near_exit = (new_road == vehicle.conn_road and
                                 vehicle.connector_end - vehicle.position <= reach_long)
                    if near_exit:
                        # Bodies that already crossed the exit seam still
                        # share the space just past the connector end.
                        nearby.extend(other for other in self._nearby(
                            vehicle.out_road, 0.0, reach_long)
                            if other.road == vehicle.out_road)
                    if new_road == vehicle.conn_road and road_progress < reach_long:
                        # A heavy vehicle of the same approach still standing
                        # at the stop line already reaches up to 4.5 m into
                        # the junction with its front.
                        nearby.extend(other for other in self.vehicles
                                      if other.wheelbase > 0.0 and other.arm == vehicle.arm and
                                      other.road == other.in_road and
                                      other.position + other.length / 2 > other.approach_end)
                    for other in nearby:
                        if other is vehicle:
                            continue
                        other_road, other_progress = self._road_state(other)
                        other_shift = other.current_lateral_shift
                        lo, hi = own_lo, own_hi
                        road_progress = own_progress
                        ego_cur, ego_next = vehicle.current_lateral_shift, next_shift
                        if near_exit and other_road == vehicle.out_road:
                            # Compare in exit-road coordinates.
                            seam_progress, cur_out = self._continuous_seam_pose(
                                new_road, other_road, vehicle.current_lateral_shift)
                            _, next_out = self._continuous_seam_pose(
                                new_road, other_road, next_shift)
                            road_progress = (vehicle.position - vehicle.connector_end +
                                             seam_progress)
                            lo, hi = min(cur_out, next_out), max(cur_out, next_out)
                            ego_cur, ego_next = cur_out, next_out
                        elif (other_road == other.in_road and other.wheelbase > 0.0 and
                              new_road == vehicle.conn_road):
                            # Compare in this connector's coordinates.
                            seam_progress, other_shift = self._continuous_seam_pose(
                                other.in_road, new_road, other_shift)
                            other_progress = (other.position - other.approach_end +
                                              seam_progress)
                        elif other_road != new_road:
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
                        needed = max(1.2, vehicle.gap, other.gap, moving_gap)
                        if vehicle.speed < 0.5 and other.speed < 0.5:
                            # Standing bodies that do not overlap lengthwise
                            # cannot meet by a sideways move. The moving-gap
                            # rule (1.2 m) exceeds a car's jam gap (1.0 m), so
                            # a queued follower froze the change half-way.
                            needed = 0.0
                        if separation - closing * remaining_change >= needed:
                            continue
                        if self._moving_away(ego_cur, ego_next,
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
        if self.mode == "safe":
            limit = (self.max_decel + 0.5) * DT
            for vehicle in survivors:
                before = starting_speeds.get(vehicle.creation_id)
                if before is not None and before - vehicle.speed > limit:
                    self.hard_brake_events += 1
        if (self.mode == "safe" and self.jam_timeout > 0
                and math.isfinite(self.jam_timeout)):
            kept = []
            for vehicle in survivors:
                if vehicle.road == vehicle.conn_road and vehicle.speed < 0.1:
                    vehicle.junction_standstill += DT
                    if vehicle.junction_standstill > self.jam_timeout:
                        self.jam_removed += 1
                        continue
                else:
                    vehicle.junction_standstill = 0.0
                kept.append(vehicle)
            survivors = kept
        self._dest_groups = None
        self.vehicles = survivors
        # The index is only valid inside this tick's movement pass.
        self._road_index = None
        self._exited_this_tick = []
        if self.mode == "safe":
            self._update_body_sweep()
            self._resolve_road_conflicts()
            self._resolve_connector_overlaps(starting_positions, starting_speeds)
            self._resolve_seam_overlaps(starting_positions, starting_speeds)
        self._commit_axles()

    @cython.annotation_typing(True)
    def _resolve_connector_overlaps(self, starting_positions, starting_speeds):
        """Stop a following connector body before it enters an occupied footprint.

        The negotiation model works with sampled centerline bands. Curved,
        offset connector lanes can still bring two rectangles together.
        This final geometry check only retreats within the distance travelled
        in the current tick and never rewinds to an earlier tick.
        """
        vehicle: Vehicle
        groups = {}
        # The 8 m window below covers two passenger cars; longer bodies
        # overlap at larger centre distances.
        slack = self._length_slack()
        for vehicle in self.vehicles:
            road, progress = self._road_state(vehicle)
            if not road.endswith(("_in", "_out")):
                groups.setdefault(road, []).append((progress, vehicle))
        for road, entries in groups.items():
            entries.sort(key=lambda item: item[0], reverse=True)
            placed = []
            def make_polygons(item):
                # Physical bodies: the corner reach (``body_sweep``) already
                # widens the lateral rules that keep vehicles apart.
                return self._body_polygons(item, road)

            def touches(item, blockers):
                own = make_polygons(item)
                return any(polygons_overlap(a, b, 0.0)
                           for polygons in blockers for a in own for b in polygons)
            for progress, vehicle in entries:
                blockers = [other for ahead, _, other in placed
                            if ahead - progress < 8.0 + slack]
                if blockers and touches(vehicle, blockers):
                    end = vehicle.position
                    start = min(end, starting_positions.get(vehicle.creation_id, end))
                    vehicle.position = start
                    if not touches(vehicle, blockers):
                        lo, hi = start, end
                        for _ in range(18):
                            mid = (lo + hi) / 2
                            vehicle.position = mid
                            if touches(vehicle, blockers):
                                hi = mid
                            else:
                                lo = mid
                        vehicle.position = lo
                    distance = vehicle.position - start
                    vehicle.speed = max(0.0, min(vehicle.speed,
                        2.0 * distance / DT - starting_speeds.get(vehicle.creation_id, 0.0)))
                placed.append((vehicle.position - vehicle.approach_end,
                               vehicle, make_polygons(vehicle)))

    @cython.annotation_typing(True)
    def _resolve_seam_overlaps(self, starting_positions, starting_speeds):
        """Stop an approach body short of a body just past the stop line,
        when one of the two is a heavy vehicle.

        ``_resolve_connector_overlaps`` compares bodies on the same connector
        only. At the stop-line seams the connector starts with a jog of up to
        ~10 degrees, so a vehicle still on the approach and the rear unit of
        a truck or bus already on the connector are turned against each
        other: a route gap of more than a metre can still let the corner of
        the follower into the truck. Like the connector guard this only
        shortens the follower's move of this tick; it never moves anybody
        back beyond its start. Pairs of light vehicles keep the original
        behaviour.
        """
        vehicle: Vehicle
        other: Vehicle
        if not any(v.wheelbase > 0.0 for v in self.vehicles):
            return
        near_seam = {}
        for other in self.vehicles:
            if (other.road == other.conn_road and
                    other.position - other.approach_end < max(other.length, 9.0) + 2.0):
                near_seam.setdefault(other.arm, []).append(other)
        for vehicle in self.vehicles:
            if (vehicle.road != vehicle.in_road or
                    vehicle.approach_end - vehicle.position > vehicle.length / 2 + 2.0):
                continue
            # Pairs with a heavy vehicle on either side; light pairs keep the
            # original behaviour.
            blockers = [other for other in near_seam.get(vehicle.arm, ())
                        if other.wheelbase > 0.0 or vehicle.wheelbase > 0.0]
            if not blockers:
                continue
            end = vehicle.position
            start = min(end, starting_positions.get(vehicle.creation_id, end))
            if end - start < 1e-9:
                continue
            bodies = [polygon for other in blockers if other.position > vehicle.position
                      for polygon in self._body_polygons(other)]

            def touches():
                return any(polygons_overlap(own, other, 0.0)
                           for own in self._body_polygons(vehicle, vehicle.in_road)
                           for other in bodies)
            if not bodies or not touches():
                continue
            vehicle.position = start
            if touches():
                vehicle.position = end  # contact from an earlier tick: leave it
                continue
            lo, hi = start, end
            for _ in range(18):
                vehicle.position = (lo + hi) / 2
                if touches():
                    hi = vehicle.position
                else:
                    lo = vehicle.position
            vehicle.position = lo
            vehicle.speed = max(0.0, min(vehicle.speed, 2.0 * (lo - start) / DT -
                                         starting_speeds.get(vehicle.creation_id, 0.0)))

    @cython.annotation_typing(True)
    def _resolve_road_conflicts(self):
        """Keep vehicle bodies and their minimum clearance apart on road links.

        Connector links form the intersection and are intentionally excluded.
        Progress is road-relative: vehicles entering the same exit from different
        arms have different cumulative route positions.
        """
        progress: cython.double
        allowed: cython.double
        ahead_progress: cython.double
        correction: cython.double
        clearance: cython.double
        slack: cython.double
        vehicle: Vehicle
        ahead: Vehicle
        by_road = {}
        # The 6 m window below is a car length plus its clearance.
        slack = self._length_slack()
        for vehicle in self.vehicles:
            road, progress = self._road_state(vehicle)
            if road.endswith(("_in", "_out")):
                by_road.setdefault(road, []).append((progress, vehicle))
        for road, entries in by_road.items():
            entries.sort(key=lambda item: item[0], reverse=True)
            settled = []
            for progress, vehicle in entries:
                allowed = progress
                # Progress decreases through this road. Once a settled body is
                # outside the forward window, it cannot constrain a later entry.
                settled = [pair for pair in settled if pair[0] - progress <= 6.0 + slack]
                for ahead_progress, ahead in reversed(settled):
                    # Physical body widths: this guard moves vehicles back,
                    # it must not act on the corner reach (``body_sweep``)
                    # that only keeps traffic clear of long bodies.
                    if abs(vehicle.current_lateral_shift - ahead.current_lateral_shift) >= (
                            vehicle.half_width + ahead.half_width + LATERAL_CLEARANCE):
                        continue
                    clearance = max(1.0, vehicle.gap, ahead.gap)
                    if vehicle.speed < 0.5:
                        # A standing merger may slot in just ahead of a
                        # standing follower (see ``_choose_lane``). Bodies
                        # that do not overlap are left where they are: a
                        # standing follower is never pushed backwards to
                        # restore the moving clearance (also not when its
                        # leader pulls away), it simply waits for the gap.
                        clearance = 0.0
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
                        connector = vehicle.conn_road
                        vehicle.lane = self._mapped_lane(
                            vehicle, vehicle.lane, road, connector)
                        shift = (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                        vehicle.road = connector
                        vehicle.current_lateral_shift = shift
                        vehicle.start_lateral_shift = shift
                        vehicle.target_lateral_shift = shift
                        vehicle.lane_change_progress = 1.0
                        vehicle.lane_change_from = -1
                    # Rounding residues (~1e-15 m) are not corrections.
                    if correction > 1e-9:
                        self.road_safety_corrections += 1
                    if correction > self.max_road_safety_correction:
                        self.largest_road_safety_event = (
                            self.cycle, road, vehicle.demand_row_id, correction)
                    self.max_road_safety_correction = max(
                        self.max_road_safety_correction, correction)
                if vehicle.road == road:
                    settled.append((allowed, vehicle))

    @cython.annotation_typing(True)
    def _choose_lane(self, vehicle: Vehicle, lane_maps):
        progress: cython.double
        stop_gap: cython.double
        old_gap: cython.double
        front_gap: cython.double
        back_gap: cython.double
        old_acc: cython.double
        gain: cython.double
        best_gain: cython.double
        min_gain: cython.double
        target_acc: cython.double
        follower_delta: cython.double
        leader_speed: cython.double
        urgency: cython.double
        cand_shift: cython.double
        lo_lat: cython.double
        hi_lat: cython.double
        long_sep: cython.double
        closing: cython.double
        jam: cython.double
        lat_gap: cython.double
        other: Vehicle
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
        old_result = self._neighbors(vehicle, vehicle.lane, lane_maps)
        old_leader, old_gap, old_follower, _ = old_result
        # On the early returns below nothing the query depends on changes,
        # so ``_move`` reuses this result for its own query of the same lane.
        reuse = (vehicle, vehicle.lane, old_result)
        overlap = old_gap < 0
        if (not mandatory and not queue_filter and not overlap and
                old_gap < vehicle.gap + vehicle.speed * vehicle.headway):
            vehicle.time_since_lane_change += DT
            self._neighbors_reuse = reuse
            return
        if not overlap and not mandatory and not queue_filter:
            if (old_gap > max(25.0, vehicle.speed * 3.0) or
                    old_leader is None or old_leader.speed >= 0.92 * vehicle.desired or
                    (road.endswith("_in") and vehicle.maneuver != "through" and
                     self._stop_position(vehicle) - vehicle.position <= 40.0)):
                vehicle.time_since_lane_change += DT
                vehicle.leading_distance = old_gap
                vehicle.leading_speed = old_leader.speed if old_leader else 0.0
                self._neighbors_reuse = reuse
                return
        if not overlap and vehicle.time_since_lane_change < (0.8 if mandatory or queue_filter else 4.0):
            vehicle.time_since_lane_change += DT
            vehicle.leading_distance = old_gap
            vehicle.leading_speed = old_leader.speed if old_leader else 0.0
            self._neighbors_reuse = reuse
            return
        old_leader_speed = old_leader.speed if old_leader else 0.0
        old_acc = (-sys.float_info.max if overlap else
                   self._idm(vehicle, old_gap, old_leader_speed))
        vehicle.leading_distance = old_gap
        vehicle.leading_speed = old_leader_speed
        best_lane, best_gain = vehicle.lane, 0.0
        best_leader, best_front_gap = old_leader, old_gap
        min_gain = 0.5 if mandatory else (1.5 if queue_filter else 0.8)
        last_lane = self.roads[road]["num_lanes"] - vehicle.occupied
        for lane in (vehicle.lane - 1, vehicle.lane + 1):
            if not 0 <= lane <= last_lane:
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
                    if follower.speed < 0.5 and vehicle.speed < 0.5:
                        # Both standing: bodies that do not overlap are enough.
                        # A follower that crept inside its own jam gap behind
                        # the claim otherwise blocks the merger for ever, and
                        # the merger, held at its buffer, blocks the approach.
                        back_floor = 0.0
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
                # Impatient mergers accept the smaller of the two jam gaps,
                # and any positive gap to a body that is standing like them.
                jam = (0.0 if vehicle.speed < 0.5 and other.speed < 0.5
                       else min(vehicle.gap, other.gap) - 0.05)
                if long_sep >= ((1.0 - urgency) * max(1.5, vehicle.gap, other.gap) +
                                urgency * jam):
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
                    # MOBIL's safety criterion limits the braking imposed on
                    # the new follower. A standing follower brakes for nobody;
                    # IDM merely reports a large negative value for a gap
                    # inside its jam distance, which vetoed every merge in
                    # front of a queue.
                    if after <= -2.5 and follower.speed >= 0.5:
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
            entry = (progress, vehicle.creation_id, vehicle)
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
            multiplier = LATERAL_DURATION_FACTOR[vehicle.kind]
            vehicle.lateral_duration = max(0.4, base_duration * multiplier)
        else:
            self._neighbors_reuse = reuse
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
            entry = (progress, vehicle.creation_id, vehicle)
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

    @cython.annotation_typing(True)
    def _road_state(self, vehicle: Vehicle):
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

    @cython.annotation_typing(True)
    def _neighbors(self, vehicle: Vehicle, lane, lane_maps, shift=None):
        front: cython.double
        back: cython.double
        front_distance: cython.double
        back_distance: cython.double
        distance: cython.double
        other_position: cython.double
        p_other: cython.double
        other: Vehicle
        gap: cython.double
        reach: cython.double
        pair_length: cython.double
        reach_len: cython.double
        position: cython.double
        candidate_shift: cython.double
        v_length: cython.double
        v_sweep: cython.double
        v_half: cython.double
        k: cython.Py_ssize_t
        start: cython.Py_ssize_t
        index: cython.Py_ssize_t
        ahead: cython.Py_ssize_t
        behind: cython.Py_ssize_t
        road, position = self._neighbor_road_state[id(vehicle)]
        candidate_shift = ((lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH
                           if shift is None else shift)
        def lateral_interacts(other, other_position=None):
            if self.mode != "safe":
                return True
            reach = self._interaction_reach(vehicle, other, other_position, position)
            if abs(candidate_shift - other.current_lateral_shift) < reach:
                return True
            # A claimed merge counts as a leader only when the merger is
            # fully ahead (same rule as ``consider`` below). A vehicle already
            # alongside keeps going so the merger can slot in behind it;
            # otherwise the two wait for each other for ever.
            if (other_position is not None and other_position - position <
                    (vehicle.length + other.length) / 2):
                return False
            return abs(candidate_shift - other.merge_claim_shift) < reach
        leader = follower = None
        front = back = math.inf
        front_distance = back_distance = math.inf
        for sublane in range(lane, lane + vehicle.occupied):
            entries = lane_maps[road][sublane]
            index = bisect.bisect_left(entries, (position, -1))
            ahead = index
            while ahead < len(entries) and (entries[ahead][2] is vehicle or
                                            not lateral_interacts(entries[ahead][2],
                                                                  entries[ahead][0])):
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
                reach = self._interaction_reach(vehicle, other, other_position, position)
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
                # ``consider`` and ``offer`` are inlined in the two scans
                # (same arithmetic, same order): they ran ~2,000 times a tick.
                entries = road_index.entries
                start = bisect.bisect_left(entries, (position - INDEX_MARGIN, -1))
                v_half, v_sweep, v_length = vehicle.half_width, vehicle.body_sweep, vehicle.length
                for k in range(start, len(entries)):
                    p_other, _, other = entries[k]
                    if p_other - position - reach_len >= front + INDEX_MARGIN:
                        break
                    if other is vehicle:
                        continue
                    other_road, other_position = _vehicle_road_state(other)
                    if other_road != road:
                        continue
                    reach = v_half + other.half_width + LATERAL_CLEARANCE
                    pair_length = (v_length + other.length) / 2
                    if v_sweep or other.body_sweep:
                        if abs(other_position - position) >= pair_length:
                            reach += v_sweep + other.body_sweep
                    if abs(candidate_shift - other.current_lateral_shift) >= reach:
                        if not (other_position - position >= pair_length
                                and abs(candidate_shift - other.merge_claim_shift) < reach):
                            continue
                    delta = other_position - position
                    if delta >= 0:
                        gap = delta - pair_length
                        if gap < front:
                            leader, front = other, gap
                    else:
                        gap = -delta - pair_length
                        if gap < back:
                            follower, back = other, gap
                for k in range(start - 1, -1, -1):
                    p_other, _, other = entries[k]
                    if position - p_other - reach_len >= back + INDEX_MARGIN:
                        break
                    if other is vehicle:
                        continue
                    other_road, other_position = _vehicle_road_state(other)
                    if other_road != road:
                        continue
                    reach = v_half + other.half_width + LATERAL_CLEARANCE
                    pair_length = (v_length + other.length) / 2
                    if v_sweep or other.body_sweep:
                        if abs(other_position - position) >= pair_length:
                            reach += v_sweep + other.body_sweep
                    if abs(candidate_shift - other.current_lateral_shift) >= reach:
                        if not (other_position - position >= pair_length
                                and abs(candidate_shift - other.merge_claim_shift) < reach):
                            continue
                    delta = other_position - position
                    if delta >= 0:
                        gap = delta - pair_length
                        if gap < front:
                            leader, front = other, gap
                    else:
                        gap = -delta - pair_length
                        if gap < back:
                            follower, back = other, gap
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
        if follower is None and road == vehicle.conn_road:
            # On a connector, GAMA's follower search crosses its source node
            # and checks vehicles approaching on the incoming road.
            source_road = vehicle.in_road
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
            if road == vehicle.in_road:
                green = self._green_for(vehicle)
                if green or (self.mode == "gama_2025_6_4" and not vehicle.gama_entered):
                    next_road = vehicle.conn_road
                else:
                    if self.mode == "gama_2025_6_4" and vehicle.gama_entered:
                        front = vehicle.approach_end - vehicle.position + vehicle.gap
                    elif self.mode == "safe":
                        front = self._stop_position(vehicle) - vehicle.position + vehicle.gap
            elif road == vehicle.conn_road:
                next_road = vehicle.out_road
            elif (self.mode == "gama_2025_6_4" and
                  road == vehicle.out_road):
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
                            reach = self._interaction_reach(
                                vehicle, other,
                                self.roads[road]["length"] + other_position, position)
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

    @cython.annotation_typing(True)
    def _idm(self, vehicle: Vehicle, gap, leader_speed):
        road_limit: cython.double
        curve_cap: cython.double
        limit: cython.double
        stop_gap: cython.double
        desired_gap: cython.double
        interaction: cython.double
        result: cython.double
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
        curve_cap = self._curve_speed_cap(vehicle)
        limit = min(vehicle.desired, road_limit, curve_cap)
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
        if vehicle.speed > curve_cap:
            # IDM's free term only eases a vehicle back to its limit; the
            # curve cap falls at ``curve_decel`` along the path, so track it
            # with the vehicle's comfortable braking.
            result = min(result, max(-vehicle.decel, (curve_cap - vehicle.speed) / DT))
        # No tyre delivers the unbounded braking IDM asks for at tiny gaps.
        result = max(result, -self.max_decel)
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
                      jam_removed=self.jam_removed,
                      heavy_turns_rerouted=self.heavy_turns_rerouted,
                      hard_brake_events=self.hard_brake_events,
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
    parser.add_argument("--heavy-ratio", type=float, default=DEFAULT_HEAVY_RATIO,
                        help="share of heavy vehicles (trucks, buses) in random arrivals")
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
        motorbike_ratio=args.motorbike_ratio, heavy_ratio=args.heavy_ratio,
        p_left=args.p_left, p_right=args.p_right,
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
    if not __package__:
        sys.path.insert(0, str(ROOT.parent))
    from python_simulation.python_sim import main as run_cli
    run_cli()
elif __package__:
    from ._native import load_if_built
    load_if_built(__name__, __file__)
