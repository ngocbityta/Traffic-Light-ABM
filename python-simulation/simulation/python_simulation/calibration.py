"""Calibrate queue-discharge behaviour against field targets.

Two measured quantities drive signal-control results more than anything
else in this model: the saturation flow of a queued approach and the
start-up lost time. :func:`measure_discharge` measures both in a controlled
experiment (one approach saturated during red, released at green, crossings
of the stop line timed); :func:`calibrate` searches car-following and
start-up parameters so that they match targets.

There are no field measurements for Le Van Luong - Hoang Dao Thuy in this
repository. Targets therefore have to be supplied by the user (for example
from a video count of saturated green phases at the site). The repository's
Nguyen Trai - Khuat Duy Tien counts can still inform the vehicle mix, see
:func:`demand_from_counts`.
"""
from __future__ import annotations

import csv
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np

try:
    from .python_sim import ARMS, DT, LeVanLuongSimulator
    from .vehicle_geometry import SUBLANE_WIDTH
except ImportError:  # flat import
    from python_sim import ARMS, DT, LeVanLuongSimulator
    from vehicle_geometry import SUBLANE_WIDTH

# Parameter space: name -> (where, kind, key, low, high)
PARAMETER_SPACE = {
    "moto_headway": ("vehicle", "motorbike", "headway", 0.3, 1.5),
    "moto_gap": ("vehicle", "motorbike", "gap", 0.2, 1.5),
    "moto_accel": ("vehicle", "motorbike", "accel", 1.0, 3.5),
    "car_headway": ("vehicle", "car", "headway", 0.6, 2.0),
    "car_gap": ("vehicle", "car", "gap", 0.5, 2.5),
    "car_accel": ("vehicle", "car", "accel", 1.0, 3.5),
    "reaction_delay": ("startup", None, "reaction_delay", 0.0, 2.0),
    "discharge_headway": ("startup", None, "discharge_headway", 0.2, 2.0),
}


def params_to_kwargs(params):
    """Flat calibration parameters -> simulator ``vehicle_params``/``startup_params``."""
    vehicle, startup = {}, {}
    for name, value in params.items():
        where, kind, key, _, _ = PARAMETER_SPACE[name]
        if where == "vehicle":
            vehicle.setdefault(kind, {})[key] = float(value)
        else:
            startup[key] = float(value)
    return {"vehicle_params": vehicle, "startup_params": startup}


def _queue_demand(path, arm, seed, until_cycle, motorbike_ratio, p_left, p_right,
                  per_tick=1, roads=None):
    rng = random.Random(seed)
    origin = ARMS.index(arm)
    num_lanes = roads[f"{arm}_in"]["num_lanes"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for cycle in range(1, until_cycle + 1):
            for _ in range(per_tick):
                draw = rng.random()
                turn = 0 if draw < p_left else 1 if draw < p_left + p_right else 2
                is_car = rng.random() < 1.0 - motorbike_ratio
                lane = rng.randrange(0, num_lanes - (2 if is_car else 1) + 1)
                speed = max(1.0, rng.gauss(50 if is_car else 40, 5))
                writer.writerow((cycle, origin, int(is_car), speed, turn, lane))


def measure_discharge(seed=1, arm="SE", motorbike_ratio=0.85, p_left=0.2, p_right=0.2,
                      fit_start=10.0, **sim_kwargs):
    """Saturated discharge of one approach in the first green after a full red.

    Vehicles arrive on ``arm`` at one per tick from t = 0 until its green
    starts, so a long queue covers the whole approach width. Returns
    ``saturation_flow`` (veh/h of green, whole approach; slope of the
    cumulative count from ``fit_start`` to the end of green), the same per
    metre of approach width, ``lost_time`` (s; ``fit_start - N(fit_start)/s``,
    the classic start-up lost time) and ``first_10s`` (crossings in the first
    10 s). A negative lost time means the first seconds discharge faster than
    the saturation rate (a front-row burst across many sublanes).
    """
    probe = LeVanLuongSimulator(seed=seed, flow_per_approach=0, **sim_kwargs)
    green_phase = "LVL" if arm in ("NE", "SW") else "HDT"
    onset = next(c for c in range(2, 4000) if probe._phase(c) == green_phase
                 and probe._phase(c - 1) != green_phase and c > 2)
    end = next(c for c in range(onset, onset + 4000) if probe._phase(c) != green_phase)
    with tempfile.TemporaryDirectory() as tmp:
        demand = Path(tmp) / "queue.csv"
        _queue_demand(demand, arm, seed, onset - 1, motorbike_ratio, p_left, p_right,
                      roads=probe.roads)
        sim = LeVanLuongSimulator(seed=seed, flow_per_approach=0, demand_file=demand,
                                  **sim_kwargs)
        on_approach = set()
        crossings = []
        while sim.cycle < end:
            sim.step()
            current = {v.creation_id for v in sim.vehicles if v.road == v.in_road}
            if sim.cycle >= onset:
                crossed = on_approach - current
                crossings.extend([(sim.cycle - onset + 1) * DT] * len(
                    [vid for vid in crossed]))
            on_approach = current
        queued_left = len(on_approach)
    times = np.sort(np.asarray(crossings, float))
    width = probe.roads[f"{arm}_in"]["num_lanes"] * SUBLANE_WIDTH
    result = {"crossings": int(len(times)), "queue_left_at_end": queued_left,
              "approach_width_m": width, "green_s": (end - onset) * DT}
    window = times[times >= fit_start]
    if len(window) < 5:
        result.update(saturation_flow=float("nan"), saturation_flow_per_m=float("nan"),
                      lost_time=float("nan"))
        return result
    counts = np.arange(len(times) - len(window) + 1, len(times) + 1)
    slope, _ = np.polyfit(window, counts, 1)
    before = len(times) - len(window)
    result.update(saturation_flow=float(slope * 3600.0),
                  saturation_flow_per_m=float(slope * 3600.0 / width),
                  lost_time=(float(fit_start - before / slope) if slope > 0 else float("nan")),
                  first_10s=int(np.sum(times <= 10.0)))
    return result


def _objective(params, targets, seeds, base_kwargs):
    kwargs = dict(base_kwargs)
    kwargs.update(params_to_kwargs(params))
    errors = []
    measured = []
    for seed in seeds:
        m = measure_discharge(seed=seed, **kwargs)
        measured.append(m)
    report = {}
    for key, target in targets.items():
        values = [m[key] for m in measured if math.isfinite(m.get(key, float("nan")))]
        if not values:
            return float("inf"), report
        mean = float(np.mean(values))
        report[key] = mean
        errors.append(((mean - target) / target) ** 2)
    return float(np.sum(errors)), report


def calibrate(targets, names=("moto_headway", "moto_gap", "reaction_delay",
                              "discharge_headway"),
              seeds=(1, 2, 3), budget=40, rng_seed=0, start=None, base_kwargs=None,
              log=None):
    """Minimise squared relative error to ``targets``.

    ``targets`` keys are outputs of :func:`measure_discharge`, e.g.
    ``{"saturation_flow_per_m": 1800, "lost_time": 2.0}`` (values from your
    own field data). Uses SciPy's bounded Nelder-Mead when available and a
    seeded random/local search otherwise; seeds are common random numbers
    across evaluations.
    """
    base_kwargs = dict(base_kwargs or {})
    names = tuple(names)
    lows = np.array([PARAMETER_SPACE[n][3] for n in names])
    highs = np.array([PARAMETER_SPACE[n][4] for n in names])
    defaults = start or {}
    x0 = np.array([defaults.get(n, _default_value(n)) for n in names], float)
    history = []

    def evaluate(x):
        x = np.clip(x, lows, highs)
        params = dict(zip(names, map(float, x)))
        loss, report = _objective(params, targets, seeds, base_kwargs)
        history.append({"params": params, "loss": loss, "measured": report})
        if log:
            log(history[-1])
        return loss

    try:
        from scipy.optimize import minimize
        # The objective is noisy (stochastic queues, common random numbers
        # only damp it), so start from a wide simplex: 25 % of each range.
        simplex = [x0] + [np.clip(x0 + np.eye(len(names))[k] * 0.25 * (highs - lows),
                                  lows, highs) for k in range(len(names))]
        result = minimize(evaluate, x0, method="Nelder-Mead",
                          bounds=list(zip(lows, highs)),
                          options={"maxfev": budget, "xatol": 1e-2, "fatol": 1e-3,
                                   "initial_simplex": np.array(simplex)})
        best_x = np.clip(result.x, lows, highs)
    except ImportError:
        rng = np.random.default_rng(rng_seed)
        best_x, best = x0, evaluate(x0)
        scale = (highs - lows) * 0.25
        for _ in range(budget - 1):
            candidate = np.clip(best_x + rng.normal(0.0, 1.0, len(names)) * scale, lows, highs)
            loss = evaluate(candidate)
            if loss < best:
                best_x, best = candidate, loss
            else:
                scale *= 0.97
    best_entry = min(history, key=lambda h: h["loss"])
    return {"best": best_entry, "kwargs": params_to_kwargs(best_entry["params"]),
            "evaluations": len(history), "history": history}


def _default_value(name):
    where, kind, key, low, high = PARAMETER_SPACE[name]
    if where == "vehicle":
        try:
            from .python_sim import DEFAULT_VEHICLE_PARAMS
        except ImportError:
            from python_sim import DEFAULT_VEHICLE_PARAMS
        return DEFAULT_VEHICLE_PARAMS[kind][key]
    return {"reaction_delay": 0.6, "discharge_headway": 0.8}[key]


def demand_from_counts(path):
    """Hourly flow and motorbike share per station from a count CSV.

    Expects the columns of ``train/benchmarks/gama/data/traffic_data_NT_KDT.csv``
    (motorbike, car, small_truck_bus, heavy_truck_bus, trailer, total_flow).
    Heavier classes are counted as "car-sized" because the simulator has two
    vehicle types.
    """
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            moto = float(row["motorbike"])
            other = sum(float(row[k]) for k in ("car", "small_truck_bus",
                                                "heavy_truck_bus", "trailer"))
            total = moto + other
            rows.append({"time_slot": row["time_slot"], "station": row["station_name"],
                         "flow_veh_h": total,
                         "motorbike_ratio": moto / total if total else float("nan")})
    return rows


def save(result, path):
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
