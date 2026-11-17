"""Paired request schedule for the Le Van Luong GAMA and Python models."""

import csv
import hashlib
import math
import random
from pathlib import Path

from .python_sim import (ARMS, DEFAULT_HEAVY_RATIO, DEFAULT_VEHICLE_PARAMS, DT,
                         VEHICLE_KINDS, kind_from_draw, occupied_sublanes,
                         road_lengths, turn_lane_bounds, vehicle_kind)

# Sublanes occupied by each vehicle type at its default width.
OCCUPIED = {kind: occupied_sublanes(DEFAULT_VEHICLE_PARAMS[kind]["width"])
            for kind in VEHICLE_KINDS}


def normalize_requested_lane(roads, arm, is_car, turn, lane):
    """Return the model's shared initial lane for a physically valid request.

    Demand files retain the requested lane. Both backends must apply this same
    turn-corridor rule when creating the vehicle. ``is_car`` is the legacy car
    flag (bool) or a vehicle type name ("motorbike", "car", "heavy").
    """
    if (arm not in ARMS or turn not in (0, 1, 2) or
            not isinstance(is_car, (bool, str)) or
            (isinstance(is_car, str) and is_car not in VEHICLE_KINDS)):
        raise ValueError("Invalid arm, vehicle type, or turn")
    road = roads[f"{arm}_in"]
    occupied = OCCUPIED[vehicle_kind(is_car)]
    maximum = road["num_lanes"] - occupied
    if isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane <= maximum:
        raise ValueError(f"Invalid requested lane {lane!r} for {arm} (valid: 0..{maximum})")
    if turn == 2:
        return lane
    minimum, maximum = turn_lane_bounds(road["num_lanes"], occupied,
                                        ("left", "right", "through")[turn])
    return min(max(lane, minimum), maximum)


def read_demand(path, roads=None):
    """Read and validate the six-column paired demand format.

    The third column is the vehicle type: 0 motorbike, 1 car, 2 heavy vehicle
    (files written before heavy vehicles existed only hold 0 and 1).
    """
    roads = road_lengths() if roads is None else roads
    rows = []
    previous = None
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for line_number, raw in enumerate(csv.reader(stream), 1):
            if not raw:
                continue
            if len(raw) != 6:
                raise ValueError(f"Demand line {line_number}: expected 6 columns, got {len(raw)}")
            try:
                cycle, origin, car = int(raw[0]), int(raw[1]), int(raw[2])
                speed, turn, lane = float(raw[3]), int(raw[4]), int(raw[5])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Demand line {line_number}: invalid numeric field") from exc
            if cycle < 1 or not 0 <= origin < len(ARMS) or not 0 <= car < len(VEHICLE_KINDS):
                raise ValueError(f"Demand line {line_number}: invalid cycle, arm, or vehicle type")
            if not math.isfinite(speed) or speed <= 0 or turn not in (0, 1, 2):
                raise ValueError(f"Demand line {line_number}: speed must be finite and positive; turn must be 0..2")
            arm = ARMS[origin]
            try:
                normalize_requested_lane(roads, arm, VEHICLE_KINDS[car], turn, lane)
            except ValueError as exc:
                raise ValueError(f"Demand line {line_number}: {exc}") from exc
            order = (cycle, origin)
            if previous is not None and order < previous:
                raise ValueError(f"Demand line {line_number}: rows must be ordered by cycle and arm")
            previous = order
            rows.append((cycle, origin, car, speed, turn, lane))
    return rows


def generate_demand(path, seed, horizon, flow_per_approach=2400.0,
                    motorbike_ratio=0.85, p_left=0.2, p_right=0.2,
                    heavy_ratio=DEFAULT_HEAVY_RATIO):
    """CSV rows: cycle, arm, vehicle type, speed in km/h, turn, initial lane.

    Vehicle type: 0 motorbike, 1 car, 2 heavy vehicle (truck or bus). Pass
    ``heavy_ratio=0`` for the two-type demand the GAMA model replays.
    """
    params = (flow_per_approach, motorbike_ratio, p_left, p_right, heavy_ratio)
    if (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0 or
            any(not math.isfinite(x) for x in params) or
            flow_per_approach < 0 or not 0 <= motorbike_ratio <= 1 or
            not 0 <= heavy_ratio <= 1 or motorbike_ratio + heavy_ratio > 1 + 1e-12 or
            p_left < 0 or p_right < 0 or p_left + p_right > 1):
        raise ValueError("Invalid demand parameters")
    roads = road_lengths()
    rng = random.Random(seed)
    rows = []
    probability = flow_per_approach / 3600.0 * DT
    if not 0 <= probability <= 1:
        raise ValueError("Arrival probability must be between zero and one")
    for cycle in range(1, horizon + 1):
        for origin, arm in enumerate(ARMS):
            if rng.random() >= probability:
                continue
            draw = rng.random()
            turn = 0 if draw < p_left else 1 if draw < p_left + p_right else 2
            kind = kind_from_draw(rng.random(), motorbike_ratio, heavy_ratio)
            occupied = OCCUPIED[kind]
            turn_name = ("left", "right", "through")[turn]
            minimum, maximum = ((0, roads[f"{arm}_in"]["num_lanes"] - occupied)
                                if turn_name == "through" else turn_lane_bounds(
                                    roads[f"{arm}_in"]["num_lanes"], occupied, turn_name))
            lane = rng.randrange(minimum, maximum + 1)
            kind_params = DEFAULT_VEHICLE_PARAMS[kind]
            speed = max(1.0, rng.gauss(kind_params["speed_mean"], kind_params["speed_sd"]))
            rows.append((cycle, origin, VEHICLE_KINDS.index(kind), speed, turn, lane))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    read_demand(path, roads)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demand_parameters(path):
    return [{"name": "demand_file_path", "type": "string",
             "value": str(Path(path).resolve())}]
