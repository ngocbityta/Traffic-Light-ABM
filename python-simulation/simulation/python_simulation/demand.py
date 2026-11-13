"""Paired request schedule for the Le Van Luong GAMA and Python models."""

import csv
import hashlib
import math
import random
from pathlib import Path

from .python_sim import ARMS, DT, road_lengths, turn_lane_bounds


def normalize_requested_lane(roads, arm, is_car, turn, lane):
    """Return the model's shared initial lane for a physically valid request.

    Demand files retain the requested lane. Both backends must apply this same
    turn-corridor rule when creating the vehicle.
    """
    if arm not in ARMS or turn not in (0, 1, 2) or not isinstance(is_car, bool):
        raise ValueError("Invalid arm, vehicle type, or turn")
    road = roads[f"{arm}_in"]
    occupied = 2 if is_car else 1
    maximum = road["num_lanes"] - occupied
    if isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane <= maximum:
        raise ValueError(f"Invalid requested lane {lane!r} for {arm} (valid: 0..{maximum})")
    if turn == 2:
        return lane
    minimum, maximum = turn_lane_bounds(road["num_lanes"], occupied,
                                        ("left", "right", "through")[turn])
    return min(max(lane, minimum), maximum)


def read_demand(path, roads=None):
    """Read and validate the six-column paired demand format."""
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
            if cycle < 1 or not 0 <= origin < len(ARMS) or car not in (0, 1):
                raise ValueError(f"Demand line {line_number}: invalid cycle, arm, or car flag")
            if not math.isfinite(speed) or speed <= 0 or turn not in (0, 1, 2):
                raise ValueError(f"Demand line {line_number}: speed must be finite and positive; turn must be 0..2")
            arm = ARMS[origin]
            try:
                normalize_requested_lane(roads, arm, bool(car), turn, lane)
            except ValueError as exc:
                raise ValueError(f"Demand line {line_number}: {exc}") from exc
            order = (cycle, origin)
            if previous is not None and order < previous:
                raise ValueError(f"Demand line {line_number}: rows must be ordered by cycle and arm")
            previous = order
            rows.append((cycle, origin, car, speed, turn, lane))
    return rows


def generate_demand(path, seed, horizon, flow_per_approach=2400.0,
                    motorbike_ratio=0.85, p_left=0.2, p_right=0.2):
    """CSV rows: cycle, arm, car flag, speed in km/h, turn, initial lane."""
    params = (flow_per_approach, motorbike_ratio, p_left, p_right)
    if (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0 or
            any(not math.isfinite(x) for x in params) or
            flow_per_approach < 0 or not 0 <= motorbike_ratio <= 1 or
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
            is_car = rng.random() < 1.0 - motorbike_ratio
            occupied = 2 if is_car else 1
            turn_name = ("left", "right", "through")[turn]
            minimum, maximum = ((0, roads[f"{arm}_in"]["num_lanes"] - occupied)
                                if turn_name == "through" else turn_lane_bounds(
                                    roads[f"{arm}_in"]["num_lanes"], occupied, turn_name))
            lane = rng.randrange(minimum, maximum + 1)
            speed = max(1.0, rng.gauss(50 if is_car else 40, 5))
            rows.append((cycle, origin, int(is_car), speed, turn, lane))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    read_demand(path, roads)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demand_parameters(path):
    return [{"name": "demand_file_path", "type": "string",
             "value": str(Path(path).resolve())}]
