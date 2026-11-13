"""Small geometry-based reservation model for the Le Van Luong junction.

The API intentionally takes GeoJSON and plain request records so it can be
used by the simulator without coupling conflict logic to its vehicle class.
Coordinates in the supplied network are metric (EPSG:32648).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segments_intersection(a, b, c, d, eps=1e-8):
    """Return local fractions for a proper/endpoint segment intersection."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    qx, qy = c[0] - a[0], c[1] - a[1]
    if abs(den) < eps:
        # Collinear overlap is represented by the closest endpoints below;
        # parallel disjoint segments are not crossings.
        if abs(qx * ry - qy * rx) >= eps:
            return None
        rr = rx * rx + ry * ry
        if rr < eps:
            return (0.0, 0.0) if _distance(a, c) < eps else None
        t0 = (qx * rx + qy * ry) / rr
        t1 = t0 + (sx * rx + sy * ry) / rr
        lo, hi = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
        if lo <= hi + eps:
            # A single representative point is sufficient to make a conflict
            # interval; the footprint expansion supplies its longitudinal size.
            px, py = a[0] + lo * rx, a[1] + lo * ry
            ss = sx * sx + sy * sy
            u = 0.0 if ss < eps else ((px-c[0])*sx+(py-c[1])*sy)/ss
            return lo, min(1.0, max(0.0, u))
        return None
    t = (qx * sy - qy * sx) / den
    u = (qx * ry - qy * rx) / den
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return min(1.0, max(0.0, t)), min(1.0, max(0.0, u))
    return None


def _project_fraction(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    if den < 1e-12:
        return 0.0
    return min(1.0, max(0.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / den))


def _closest_segments(a, b, c, d):
    """Return closest segment fractions and their separation in 2D."""
    hit = _segments_intersection(a, b, c, d)
    if hit is not None:
        return hit[0], hit[1], 0.0
    candidates = []
    for p, t, x, y, flip in ((a, 0.0, c, d, False), (b, 1.0, c, d, False),
                             (c, 0.0, a, b, True), (d, 1.0, a, b, True)):
        u = _project_fraction(p, x, y)
        q = (x[0] + u * (y[0]-x[0]), x[1] + u * (y[1]-x[1]))
        dist = _distance(p, q)
        candidates.append((dist, (u, t) if flip else (t, u)))
    dist, (t, u) = min(candidates, key=lambda item: item[0])
    return t, u, dist


def _cumulative(points):
    out = [0.0]
    for a, b in zip(points, points[1:]):
        out.append(out[-1] + _distance(a, b))
    return out


@dataclass(frozen=True)
class Movement:
    id: str
    arm: str
    to_arm: str
    turn: str
    points: tuple[tuple[float, float], ...]
    lengths: tuple[float, ...]

    @property
    def length(self):
        return self.lengths[-1] if self.lengths else 0.0


@dataclass(frozen=True)
class ConflictInterval:
    """Arc-length occupancy envelope on each of two movement paths."""
    start_a: float
    end_a: float
    start_b: float
    end_b: float


class ConflictMap:
    """Precomputed movement paths and actual centerline crossing intervals."""

    def __init__(self, movements: Mapping[str, Movement], clearance=1.0,
                 default_vehicle_length=4.5, near_miss_distance=2.2):
        self.movements = dict(movements)
        self.clearance = float(clearance)
        self.default_vehicle_length = float(default_vehicle_length)
        # Approximate combined half-width plus lateral clearance for two cars.
        # Pairs added only by this envelope are exposed to downstream footprint
        # validation so callers do not mistake a centerline miss for proof of safety.
        self.near_miss_distance = float(near_miss_distance)
        self.conflicts: dict[tuple[str, str], tuple[ConflictInterval, ...]] = {}
        self.near_miss_pairs: set[tuple[str, str]] = set()
        ids = sorted(self.movements)
        for i, aid in enumerate(ids):
            for bid in ids[i + 1:]:
                intervals, near_miss = self._pair(self.movements[aid], self.movements[bid])
                if intervals:
                    self.conflicts[(aid, bid)] = tuple(intervals)
                if near_miss:
                    self.near_miss_pairs.add((aid, bid))

    @classmethod
    def from_geojson(cls, source, **kwargs):
        if isinstance(source, Mapping):
            data = source
        else:
            with open(source, encoding="utf-8") as stream:
                data = json.load(stream)
        movements = {}
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            if props.get("kind") != "connector":
                continue
            coords = feature.get("geometry", {}).get("coordinates", [])
            points = tuple((float(p[0]), float(p[1])) for p in coords)
            if len(points) < 2:
                continue
            mid = str(props["id"])
            movements[mid] = Movement(mid, str(props.get("arm", "")),
                                      str(props.get("to_arm", "")),
                                      str(props.get("turn", "through")),
                                      points, tuple(_cumulative(points)))
        return cls(movements, **kwargs)

    def _pair(self, a, b):
        found = []
        near_miss = False
        intersects = False
        for i, (a0, a1) in enumerate(zip(a.points, a.points[1:])):
            al = a.lengths[i + 1] - a.lengths[i]
            for j, (b0, b1) in enumerate(zip(b.points, b.points[1:])):
                t, u, separation = _closest_segments(a0, a1, b0, b1)
                if separation > self.near_miss_distance:
                    continue
                if separation > 1e-8:
                    near_miss = True
                else:
                    intersects = True
                sa = a.lengths[i] + t * al
                sb = b.lengths[j] + u * (b.lengths[j + 1] - b.lengths[j])
                # A vehicle occupies the conflict before its reference point
                # arrives and until its rear clears it. This symmetric envelope
                # makes pair tests independent of vehicle type.
                radius = self.default_vehicle_length / 2 + self.clearance
                found.append(ConflictInterval(max(0.0, sa-radius), min(a.length, sa+radius),
                                              max(0.0, sb-radius), min(b.length, sb+radius)))
        return _merge_intervals(found), near_miss and not intersects

    def intervals(self, movement_a, movement_b):
        if movement_a == movement_b:
            m = self.movements[movement_a]
            r = self.default_vehicle_length / 2 + self.clearance
            return (ConflictInterval(0, m.length, 0, m.length),)
        key = tuple(sorted((movement_a, movement_b)))
        values = self.conflicts.get(key, ())
        if movement_a == key[0]:
            return values
        return tuple(ConflictInterval(x.start_b, x.end_b, x.start_a, x.end_a) for x in values)


def _merge_intervals(items):
    if not items:
        return ()
    items = sorted(items, key=lambda x: (x.start_a, x.start_b))
    result = [items[0]]
    for x in items[1:]:
        p = result[-1]
        if x.start_a <= p.end_a + 0.5 and x.start_b <= p.end_b + 0.5:
            result[-1] = ConflictInterval(p.start_a, max(p.end_a, x.end_a),
                                          min(p.start_b, x.start_b), max(p.end_b, x.end_b))
        else:
            result.append(x)
    return tuple(result)


@dataclass(frozen=True)
class MovementRequest:
    vehicle_id: str
    movement_id: str
    approach_distance: float = 0.0
    speed: float = 5.0
    vehicle_length: float = 4.5
    turn: str | None = None
    committed: bool = False
    connector_s: float | None = None
    downstream_clear: bool = True
    arrival_time: float | None = None
    waiting_since: float | None = None
    # Optional kinematic description. When all four are given, conflict windows
    # are predicted with an acceleration profile (earliest arrival at full
    # acceleration, latest clearing at half acceleration) instead of a single
    # constant speed. ``entry_distance`` is the distance from the vehicle's
    # reference point to the start of its connector (0 once on the connector).
    accel: float | None = None
    desired_speed: float | None = None
    speed0: float | None = None
    entry_distance: float | None = None

    def eta(self):
        if self.arrival_time is not None:
            return float(self.arrival_time)
        return max(0.0, self.approach_distance) / max(0.1, self.speed)

    def arrival_at(self, now):
        """Absolute predicted entry time, respecting an explicit absolute ETA."""
        return float(self.arrival_time) if self.arrival_time is not None else now + self.eta()


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    blockers: tuple[str, ...] = ()


@dataclass
class Reservation:
    request: MovementRequest
    entered_at: float
    conflict_windows: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    clears_at: float = 0.0


class ConflictEngine:
    """Deterministic time-window reservations for incoming movement requests.

    Call :meth:`decide_batch` once per simulation step for all eligible requests,
    and call :meth:`release` after a vehicle's rear clears the junction.
    """

    def __init__(self, conflict_map: ConflictMap, clearance_time=0.5,
                 following_headway=0.8, left_wait_priority_after=15.0):
        self.map = conflict_map
        self.clearance_time = float(clearance_time)
        self.following_headway = float(following_headway)
        self.left_wait_priority_after = float(left_wait_priority_after)
        self.reservations: dict[str, Reservation] = {}

    def _turn(self, req):
        return req.turn or self.map.movements[req.movement_id].turn

    def _rank(self, req, now):
        # Committed traffic must be allowed to clear before newly approaching
        # traffic. Long-waiting vehicles on any movement (left, through, or right)
        # age into priority ordered FIFO by waiting_since to prevent starvation;
        # otherwise ETA and turn priority break ties.
        m = self.map.movements.get(req.movement_id)
        committed_rank = 0 if req.committed or req.connector_s is not None else 1
        turn = req.turn or (m.turn if m is not None else "left")
        wait_age = max(0.0, now - req.waiting_since) if req.waiting_since is not None else 0.0
        aged_rank = -1 if wait_age >= self.left_wait_priority_after else 0
        turn_rank = 0 if turn == "through" else (1 if turn == "right" else 2)
        if aged_rank < 0:
            return (committed_rank, aged_rank, float(req.waiting_since or 0.0), req.eta(), turn_rank, str(req.vehicle_id))
        return (committed_rank, aged_rank, req.eta(), turn_rank, str(req.vehicle_id))

    @staticmethod
    def _overlap(a, b):
        return a[0] < b[1] - 1e-9 and b[0] < a[1] - 1e-9

    @staticmethod
    def _same_corridor(mov_a: str, mov_b: str) -> bool:
        if mov_a == mov_b:
            return True
        if "@" in mov_a and "@" in mov_b:
            return mov_a.split("@", 1)[0] == mov_b.split("@", 1)[0]
        return False

    @staticmethod
    def _travel_time(distance, v0, accel, vmax):
        """Time to cover ``distance`` from speed ``v0`` accelerating up to ``vmax``."""
        if distance <= 0.0:
            return 0.0
        v0 = max(0.0, v0)
        vmax = max(vmax, v0, 0.1)
        if accel <= 1e-9 or v0 >= vmax:
            return distance / max(v0, 0.1)
        d_acc = (vmax * vmax - v0 * v0) / (2.0 * accel)
        if distance <= d_acc:
            return (-v0 + math.sqrt(v0 * v0 + 2.0 * accel * distance)) / accel
        return (vmax - v0) / accel + (distance - d_acc) / vmax

    def _candidate_foes(self, movement_id):
        """Movements that may conflict with ``movement_id``.

        A map may carry an ``adjacent`` index (movement -> conflicting
        movements); otherwise every movement is checked. Results are equal
        because ``intervals`` is empty for non-adjacent pairs.
        """
        adjacent = getattr(self.map, "adjacent", None)
        if adjacent is None:
            return self.map.movements
        return adjacent.get(movement_id, ())

    def _kinematic_windows(self, req, now):
        m = self.map.movements[req.movement_id]
        v0 = max(0.0, req.speed0)
        vmax = max(req.desired_speed, v0)
        a_hi = max(0.1, req.accel)
        a_lo = 0.5 * a_hi
        lead = max(0.0, req.entry_distance)
        pos = req.connector_s or 0.0
        length = req.vehicle_length

        def late_end(s):
            dist = lead + max(0.0, s - pos)
            t = self._travel_time(dist, v0, a_lo, vmax)
            v_end = min(vmax, math.sqrt(v0 * v0 + 2.0 * a_lo * dist))
            return now + t + length / max(v_end, 2.0) + self.clearance_time

        intervals = {}
        for other in self._candidate_foes(req.movement_id):
            if other == req.movement_id:
                continue
            windows = []
            for c in self.map.intervals(req.movement_id, other):
                start = now + self._travel_time(
                    lead + max(0.0, c.start_a - pos), v0, a_hi, vmax)
                windows.append((start, late_end(c.end_a)))
            if windows:
                intervals[other] = tuple(windows)
        return intervals, late_end(m.length)

    def _windows(self, req, now):
        if (req.accel is not None and req.desired_speed is not None and
                req.speed0 is not None and req.entry_distance is not None):
            return self._kinematic_windows(req, now)
        m = self.map.movements[req.movement_id]
        speed = max(0.1, req.speed)
        base = now if req.connector_s is not None else req.arrival_at(now)
        pos = req.connector_s or 0.0
        intervals = {}
        for other in self._candidate_foes(req.movement_id):
            if other == req.movement_id:
                continue
            windows = []
            for c in self.map.intervals(req.movement_id, other):
                start = base + max(0.0, c.start_a - pos) / speed
                end = base + max(0.0, c.end_a - pos) / speed + req.vehicle_length / speed + self.clearance_time
                windows.append((start, end))
            if windows:
                intervals[other] = tuple(windows)
        return intervals, base + max(0.0, m.length - pos) / speed + req.vehicle_length / speed + self.clearance_time

    def refresh(self, request: MovementRequest, now=0.0):
        """Refresh an existing reservation with current connector progress/speed.

        Submit the vehicle again with ``connector_s`` each simulation step, or
        call this explicitly. Refreshing updates its remaining foe windows and
        clears time instead of keeping the stale entry-time prediction.
        """
        if request.vehicle_id not in self.reservations:
            return False
        if request.movement_id not in self.map.movements:
            return False
        windows, clears_at = self._windows(request, now)
        self.reservations[request.vehicle_id] = Reservation(request, now, windows, clears_at)
        return True

    def decide_batch(self, requests: Iterable[MovementRequest], now=0.0,
                     ) -> dict[str, Decision]:
        """Rebuild all approach grants atomically from this tick's complete batch.

        Only reservations for vehicles already on connectors persist between
        calls. Unsubmitted or blocked approaches relinquish their prior grants.
        """
        requests = tuple(requests)
        # Approach grants are tick-local predictions. Retaining one from a
        # previous batch would let an absent or stale ETA outrank live traffic.
        for vehicle_id, reservation in tuple(self.reservations.items()):
            if reservation.request.connector_s is None:
                self.release(vehicle_id)
        ordered = sorted(requests, key=lambda r: self._rank(r, now))
        decisions = {}
        provisional: list[Reservation] = []
        starved_yielding: list[MovementRequest] = []
        for req in ordered:
            if req.vehicle_id in self.reservations:
                # Only committed connector reservations survive batch reset.
                self.refresh(req, now)
                decisions[req.vehicle_id] = Decision(True, "already_reserved")
                continue
            movement = self.map.movements.get(req.movement_id)
            if movement is None:
                decisions[req.vehicle_id] = Decision(False, "unknown_movement")
                continue
            # A vehicle that is committed (past the stop line, holding an entry
            # token, or in a dilemma hold) will cross whatever the arbiter says.
            # It must therefore always keep its reservation visible to later
            # requests; denying it would hide its window and let a conflicting
            # vehicle be granted onto a crossing path.
            committed_now = req.connector_s is not None or req.committed
            if not committed_now and not req.downstream_clear:
                decisions[req.vehicle_id] = Decision(False, "downstream_occupied", (movement.to_arm,))
                continue
            windows, clears_at = self._windows(req, now)
            if committed_now:
                # Once committed, a reservation conflict cannot revoke the
                # vehicle's clearance path. Keep its live window visible to
                # subsequent approach requests.
                provisional.append(Reservation(req, now, windows, clears_at))
                decisions[req.vehicle_id] = Decision(True, "committed_clearance")
                continue
            yielding_to = tuple(sorted(
                s.vehicle_id for s in starved_yielding
                if (not self._same_corridor(s.movement_id, req.movement_id)
                    and self.map.intervals(req.movement_id, s.movement_id))
            ))
            if yielding_to:
                decisions[req.vehicle_id] = Decision(False, "yielding_to_starved", yielding_to)
                continue
            blockers = []
            for existing in (*self.reservations.values(), *provisional):
                old = existing.request
                if old.vehicle_id == req.vehicle_id:
                    continue
                # Same movement (including overlapping sublane bundles on through/left
                # directional corridors) forms a car-following platoon rather than a
                # full-connector crossing conflict. Tight-radius right turns keep
                # full connector clearance between distinct sublane bundles so longer
                # vehicles do not clip curved leaders on the inside arc.
                if self._same_corridor(old.movement_id, req.movement_id):
                    same_nonright_bundle = (
                        self._turn(req) != "right"
                        and bool(self.map.intervals(req.movement_id, old.movement_id))
                    )
                    if old.movement_id == req.movement_id or same_nonright_bundle:
                        old_arrival = (old.arrival_time if old.arrival_time is not None
                                       else existing.entered_at if old.connector_s is not None
                                       else existing.entered_at + old.eta())
                        new_arrival = req.arrival_at(now)
                        if abs(new_arrival - old_arrival) < self.following_headway:
                            blockers.append(old.vehicle_id)
                    if old.movement_id == req.movement_id or self._turn(req) != "right":
                        continue
                old_windows = existing.conflict_windows
                for win in windows.get(old.movement_id, ()):
                    for old_win in old_windows.get(req.movement_id, ()):
                        if self._overlap(win, old_win):
                            blockers.append(old.vehicle_id)
                            break
                    if old.vehicle_id in blockers:
                        break
            if blockers:
                decisions[req.vehicle_id] = Decision(False, "conflict_reserved", tuple(sorted(set(blockers))))
                wait_age = max(0.0, now - req.waiting_since) if req.waiting_since is not None else 0.0
                if wait_age >= self.left_wait_priority_after and req.approach_distance <= 3.0:
                    starved_yielding.append(req)
                continue
            res = Reservation(req, now, windows, clears_at)
            provisional.append(res)
            decisions[req.vehicle_id] = Decision(True, "reserved")
        self.reservations.update((r.request.vehicle_id, r) for r in provisional)
        return decisions

    def release(self, vehicle_id):
        """Release reservation once the vehicle's rear has cleared the junction."""
        self.reservations.pop(str(vehicle_id), None)

    def prune(self, now):
        """Drop expired reservations; active reservations should normally use release()."""
        for vehicle_id, reservation in tuple(self.reservations.items()):
            if reservation.clears_at <= now:
                self.release(vehicle_id)

