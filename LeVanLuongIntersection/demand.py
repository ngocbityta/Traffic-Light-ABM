"""Paired request schedule for the Le Van Luong GAMA and Python models."""

import csv
import hashlib
import random
from pathlib import Path

from .python_sim import ARMS, DT, road_lengths


def generate_demand(path, seed, horizon, flow_per_approach=2400.0,
                    motorbike_ratio=0.85, p_left=0.2, p_right=0.2):
    """CSV rows: cycle, arm, car flag, speed in km/h, turn, initial lane."""
    if (horizon < 0 or not 0 <= motorbike_ratio <= 1 or
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
            lane = rng.randrange(roads[f"{arm}_in"]["num_lanes"] - occupied + 1)
            speed = max(1.0, rng.gauss(50 if is_car else 40, 5))
            rows.append((cycle, origin, int(is_car), speed, turn, lane))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demand_parameters(path):
    return [{"name": "demand_file_path", "type": "string",
             "value": str(Path(path).resolve())}]
