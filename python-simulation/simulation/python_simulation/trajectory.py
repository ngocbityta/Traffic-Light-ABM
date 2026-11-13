"""Per-tick trajectory recording for offline surrogate-safety analysis.

The recorder stores one row per vehicle per recorded tick in growable numpy
columns (no pandas/pyarrow needed) and saves them as a compressed ``.npz``.
Analysis lives in :mod:`python_simulation.ssm`; keeping it offline means the
simulation loop pays only for a pose evaluation per vehicle.

Columns (all 1-D, aligned):
    cycle, vid, road, progress, shift, x, y, heading, speed, accel, length,
    half_width, is_car, movement, arm, maneuver
``road``/``movement``/``arm``/``maneuver`` are integer codes; the code tables
are stored alongside (``road_names``, ``movement_names``...). ``movement`` is
the lane-level connector movement id while the vehicle is on a connector and
-1 elsewhere. Signal state per recorded tick is stored in ``tick_cycle``,
``tick_phase`` and ``tick_yellow``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    from .python_sim import ARMS, DT
except ImportError:  # flat import
    from python_sim import ARMS, DT

FLOAT_COLUMNS = ("progress", "shift", "x", "y", "heading", "speed", "accel",
                 "length", "half_width")
INT_COLUMNS = ("cycle", "vid", "road", "is_car", "movement", "arm", "maneuver")
MANEUVERS = ("left", "right", "through")
PHASES = ("LVL", "HDT", "none")


class _Column:
    __slots__ = ("data", "size")

    def __init__(self, dtype, capacity=4096):
        self.data = np.empty(capacity, dtype=dtype)
        self.size = 0

    def extend(self, values):
        n = len(values)
        if self.size + n > len(self.data):
            grown = np.empty(max(2 * len(self.data), self.size + n), dtype=self.data.dtype)
            grown[:self.size] = self.data[:self.size]
            self.data = grown
        self.data[self.size:self.size + n] = values
        self.size += n

    def view(self):
        return self.data[:self.size]


class TrajectoryRecorder:
    """Record a :class:`LeVanLuongSimulator` after each ``step``.

    >>> rec = TrajectoryRecorder(sim)
    >>> for _ in range(1000):
    ...     sim.step(); rec.record()
    >>> data = rec.arrays()          # dict of numpy arrays
    >>> rec.save("run.npz")
    """

    def __init__(self, sim):
        self.sim = sim
        self.road_names = sorted(sim.roads)
        self._road_code = {name: i for i, name in enumerate(self.road_names)}
        self.movement_names = sorted(sim._lane_movement_defs)
        self._movement_code = {name: i for i, name in enumerate(self.movement_names)}
        self._float = {name: _Column(np.float64) for name in FLOAT_COLUMNS}
        self._int = {name: _Column(np.int64) for name in INT_COLUMNS}
        self._ticks = {"tick_cycle": _Column(np.int64), "tick_phase": _Column(np.int64),
                       "tick_yellow": _Column(np.int64)}
        self._last_speed = {}

    def record(self):
        sim = self.sim
        rows = {name: [] for name in FLOAT_COLUMNS + INT_COLUMNS}
        last_speed = {}
        for v in sim.vehicles:
            road, progress = sim._road_state(v)
            geometry = sim._geometry_roads[road]
            pose = geometry.pose(progress, v.current_lateral_shift)
            previous = self._last_speed.get(v.creation_id)
            accel = 0.0 if previous is None else (v.speed - previous) / DT
            last_speed[v.creation_id] = v.speed
            on_connector = road == f"{v.arm}_{v.to_arm}"
            movement = (self._movement_code.get(sim._lane_movement_id(v, on_connector=True), -1)
                        if on_connector else -1)
            rows["progress"].append(progress)
            rows["shift"].append(v.current_lateral_shift)
            rows["x"].append(pose.x)
            rows["y"].append(pose.y)
            rows["heading"].append(pose.heading)
            rows["speed"].append(v.speed)
            rows["accel"].append(accel)
            rows["length"].append(v.length)
            rows["half_width"].append(sim._half_width(v))
            rows["cycle"].append(sim.cycle)
            rows["vid"].append(v.creation_id)
            rows["road"].append(self._road_code[road])
            rows["is_car"].append(1 if v.occupied == 2 else 0)
            rows["movement"].append(movement)
            rows["arm"].append(ARMS.index(v.arm))
            rows["maneuver"].append(MANEUVERS.index(v.maneuver))
        self._last_speed = last_speed
        for name in FLOAT_COLUMNS:
            self._float[name].extend(rows[name])
        for name in INT_COLUMNS:
            self._int[name].extend(rows[name])
        self._ticks["tick_cycle"].extend([sim.cycle])
        self._ticks["tick_phase"].extend([PHASES.index(sim.phase)])
        yellow = getattr(sim, "yellow_phase", None)
        self._ticks["tick_yellow"].extend([PHASES.index(yellow) if yellow else -1])

    def arrays(self):
        out = {name: col.view().copy() for name, col in self._float.items()}
        out.update({name: col.view().copy() for name, col in self._int.items()})
        out.update({name: col.view().copy() for name, col in self._ticks.items()})
        out["road_names"] = np.array(self.road_names)
        out["road_lengths"] = np.array([self.sim.roads[name]["length"]
                                        for name in self.road_names])
        out["movement_names"] = np.array(self.movement_names)
        out["dt"] = np.array(DT)
        return out

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return path


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def record_run(sim, ticks, every=1):
    """Step ``sim`` for ``ticks`` ticks, recording every ``every`` ticks."""
    recorder = TrajectoryRecorder(sim)
    for k in range(int(ticks)):
        sim.step()
        if (k + 1) % every == 0:
            recorder.record()
    return recorder


def movement_conflict_points(sim, recorder):
    """Crossing points between lane-level connector movements.

    Returns ``{(code_a, code_b): [(s_a, s_b), ...]}`` (``code_a < code_b``)
    where the two lane centrelines cross or come closer than the sum of the
    widest half-widths (1.7 m); ``s`` are arc lengths along each movement.
    Movements from the same approach are skipped: they leave side by side in
    different sublanes (car-following/lateral interaction, covered by the
    rear-end measures), not a crossing.
    """
    try:
        from .junction_conflicts import _closest_segments
    except ImportError:
        from junction_conflicts import _closest_segments
    names = recorder.movement_names if hasattr(recorder, "movement_names") else recorder
    defs = sim._lane_movement_defs
    out = {}
    for i, a_name in enumerate(names):
        a = defs[a_name]
        for j in range(i + 1, len(names)):
            b_name = names[j]
            if a_name.split("_")[0] == b_name.split("_")[0]:
                # Same origin arm: the paths start side by side in different
                # sublanes (diverge); their interaction is lateral/rear-end,
                # not a crossing, and would produce spurious negative PETs.
                continue
            b = defs[b_name]
            # Bodies overlap laterally when centrelines are closer than the
            # sum of the occupants' half-widths (car 0.85 m, motorbike 0.35 m).
            reach = sum(0.85 if name.endswith(":2") else 0.35 for name in (a_name, b_name))
            points = []
            for ia, (a0, a1) in enumerate(zip(a.points, a.points[1:])):
                la = a.lengths[ia + 1] - a.lengths[ia]
                for ib, (b0, b1) in enumerate(zip(b.points, b.points[1:])):
                    t, u, separation = _closest_segments(a0, a1, b0, b1)
                    if separation <= reach:
                        points.append((a.lengths[ia] + t * la,
                                       b.lengths[ib] + u * (b.lengths[ib + 1] - b.lengths[ib])))
            if points:
                # Keep one representative per 2 m cluster along movement a.
                points.sort()
                merged = [points[0]]
                for p in points[1:]:
                    if p[0] - merged[-1][0] > 2.0:
                        merged.append(p)
                out[(i, j)] = merged
    return out
