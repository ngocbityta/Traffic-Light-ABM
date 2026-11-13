"""SUMO-inspired negotiation of crossing movements inside a two-phase junction.

This module replaces a central "reserve the whole connector" arbiter with the
two-layer logic SUMO uses for unprotected movements (``MSLink::opened`` /
``blockedByFoe`` before the junction and ``MSLink::getLeaderInfo`` +
``MSVehicle::isLeader`` inside it), adapted to a sublane/non-lane-based layout:

* Conflict *zones* are computed per pair of **sublane paths** (one path per
  connector, lateral sublane and vehicle width). Two paths only conflict where
  their centre lines come closer than the half widths plus a lateral margin, so
  a motorbike whose sublane path passes beside a waiting group is not blocked.
  This is what lets two-wheelers *filter* (lach) through the conflict area.
* Each tick every pair of vehicles that share a zone gets an *order* (who goes
  first). The second vehicle receives a stop target just before its zone entry
  and approaches it with the normal car-following law; it may still "advance"
  (SUMO ``jmAdvance``) when it cannot arrive before the first has cleared.
* Right of way follows the turn rank (through > right > left, SUMO major/minor
  links). A minor vehicle may take the zone first when the major one can still
  brake comfortably and arrives late enough (SUMO ``jmTimegapMinor``). SUMO's
  ``impatience`` shifts the major's arrival towards its braking arrival
  (``computeFoeArrivalTimeBraking``); here impatience also grows with group
  size, following Vu & Shimizu (2010): the size of a left-turn motorbike group
  makes its leaders more aggressive. This produces forcing (chen ngang) and
  group crossing as emergent behaviour, while the major vehicle still reacts
  physically because it then yields at its own zone entry.
* Vehicles already inside a zone, vehicles that can no longer stop, and
  vehicles clearing the junction after their phase ended go first.

All quantities are metres/seconds. Path coordinates ``s`` are connector
progress; vehicles still on the approach have negative ``front`` values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Iterable

import numpy as np

TURN_RANK = {"through": 0, "right": 1, "left": 2}
INF = math.inf


@dataclass(frozen=True)
class Zone:
    a_in: float
    a_out: float
    b_in: float
    b_out: float
    kind: str  # "cross", "merge" (same exit) or "diverge" (same entry)

    def swapped(self):
        return Zone(self.b_in, self.b_out, self.a_in, self.a_out, self.kind)


@dataclass(frozen=True)
class PathSample:
    points: np.ndarray  # (n, 2)
    s: np.ndarray       # (n,)
    half_width: float
    length: float


class ZoneMap:
    """Lazily computed conflict zones between sublane paths.

    ``sampler(movement_id)`` must return a :class:`PathSample`. Zones are the
    contiguous stretches where the two centre lines are closer than
    ``half_width_a + half_width_b + lateral_margin``.
    """

    def __init__(self, sampler: Callable[[str], PathSample],
                 lateral_margin=0.2, split_gap=1.5):
        self.sampler = sampler
        self.lateral_margin = float(lateral_margin)
        self.split_gap = float(split_gap)
        self._cache: dict[tuple[str, str], tuple[Zone, ...]] = {}
        self._samples: dict[str, PathSample] = {}

    def sample(self, movement):
        if movement not in self._samples:
            self._samples[movement] = self.sampler(movement)
        return self._samples[movement]

    def zones(self, a: str, b: str) -> tuple[Zone, ...]:
        if a == b:
            return ()
        key = (a, b) if a < b else (b, a)
        if key not in self._cache:
            self._cache[key] = self._compute(*key)
        zones = self._cache[key]
        return zones if key[0] == a else tuple(z.swapped() for z in zones)

    def _compute(self, a, b):
        pa, pb = self.sample(a), self.sample(b)
        thr = pa.half_width + pb.half_width + self.lateral_margin
        lo_a, hi_a = pa.points.min(0) - thr, pa.points.max(0) + thr
        lo_b, hi_b = pb.points.min(0), pb.points.max(0)
        if (hi_a < lo_b).any() or (hi_b < lo_a).any():
            return ()
        diff = pa.points[:, None, :] - pb.points[None, :, :]
        close = (diff * diff).sum(-1) < thr * thr
        rows = np.flatnonzero(close.any(1))
        if rows.size == 0:
            return ()
        step_a = pa.s[1] - pa.s[0] if pa.s.size > 1 else 1.0
        breaks = np.flatnonzero(np.diff(rows) * step_a > self.split_gap)
        runs = np.split(rows, breaks + 1)
        out = []
        for run in runs:
            cols = np.flatnonzero(close[run].any(0))
            a_in, a_out = float(pa.s[run[0]]), float(pa.s[run[-1]])
            b_in, b_out = float(pb.s[cols.min()]), float(pb.s[cols.max()])
            eps_a = 1.5 * step_a
            eps_b = 1.5 * (pb.s[1] - pb.s[0] if pb.s.size > 1 else 1.0)
            if a_in <= eps_a and b_in <= eps_b:
                kind = "diverge"
            elif a_out >= pa.length - eps_a and b_out >= pb.length - eps_b:
                kind = "merge"
            else:
                kind = "cross"
            out.append(Zone(a_in, a_out, b_in, b_out, kind))
        return tuple(out)


@dataclass
class Agent:
    vid: str
    movement: str          # sublane path id, e.g. "NE_SW@3:1"
    arm: str
    turn: str
    front: float           # front position along the path (connector progress)
    length: float
    speed: float
    accel: float
    decel: float           # comfortable deceleration
    max_decel: float       # emergency deceleration
    min_gap: float
    desired: float
    motorbike: bool
    on_junction: bool      # front has crossed the stop line
    arm_green: bool
    wait_time: float = 0.0
    group_size: int = 1
    road: str = ""
    hold_time: float = 0.0  # time held by a junction stop target at low speed
    commit_time: float | None = None  # passed its wait point (SUMO conflict entry)
    passive: bool = False  # red-arm approach vehicle: only seam following
    stalled: bool = False  # stopped behind its own leader: "will not pass" (SUMO willPass)

    @property
    def back(self):
        return self.front - self.length

    @property
    def corridor(self):
        return self.movement.replace("#", "@").split("@", 1)[0]


@dataclass
class NegotiationParams:
    # SUMO jmTimegapMinor (default 1 s). Two-wheelers accept tighter gaps.
    timegap_car: float = 1.0
    timegap_moto: float = 0.5
    # Minimum post-encroachment time when passing *behind* a foe.
    follow_gap_car: float = 0.5
    follow_gap_moto: float = 0.3
    # SUMO vType impatience + waiting-time growth (--time-to-impatience).
    impatience_base_car: float = 0.0
    impatience_base_moto: float = 0.2
    time_to_impatience_car: float = 30.0
    time_to_impatience_moto: float = 40.0
    # Group pressure (Vu & Shimizu 2010): extra impatience per extra member.
    group_gain: float = 0.1
    group_cap: float = 0.5
    impatience_max: float = 0.9
    # Starvation guard: a green vehicle held at a conflict for this long gets
    # maximum safe impatience and aged priority against vehicles that can stop.
    green_priority_after: float = 4.0
    # Fraction of max_decel a major vehicle is expected to use when forced.
    forced_decel_share: float = 1.0
    # Patience of a held major vehicle before it refuses to be forced again
    # (disabled by default: it starved left-turn groups in dense through flow).
    max_yield_car: float = math.inf
    max_yield_moto: float = math.inf
    # A held through vehicle lets creeping left turners cut in for this long
    # (s) before it pushes into the conflict zone itself.
    reclaim_after_car: float = 3.0
    reclaim_after_moto: float = 2.0
    # Longitudinal bunching (m) for a follower to join a crossing group.
    group_gap: float = 2.5
    # Storage for minors waiting inside the junction, per approach, in
    # motorbike equivalents (car = 2). Further minors wait at the stop line.
    wait_storage: int = 10**6
    # Treat stopped+held vehicles as not passing (True) or only those stopped
    # behind their own leader (False).
    idle_rule: bool = False
    # New-phase vehicles may slip through gaps among leftovers (lach doan).
    weave: bool = True


def travel_time(distance, v0, accel, vmax):
    """Time to cover ``distance`` from ``v0`` accelerating at ``accel`` to ``vmax``."""
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


def braking_arrival(distance, v0, decel):
    """Arrival time when braking at ``decel``; ``inf`` if the vehicle can stop."""
    if distance <= 0.0:
        return 0.0
    if decel <= 1e-9:
        return distance / max(v0, 0.1)
    disc = v0 * v0 - 2.0 * decel * distance
    if disc <= 0.0:
        return INF
    return (v0 - math.sqrt(disc)) / decel


@dataclass
class PairDecision:
    first: str
    second: str
    zone_second: Zone | None
    reason: str


class JunctionNegotiator:
    """SUMO-structured negotiation on sublane conflict zones.

    *Majors* (through/right on green) never wait for a gap: like SUMO's
    prioritised links they only react to vehicles that are physically in the
    way or that have already committed to crossing (SUMO link leaders).
    *Minors* (left turns on green, SUMO ``g`` links with an internal junction)
    drive up to a single wait point in front of their first conflict and commit
    to the whole crossing only when :meth:`minor_blocked` (SUMO ``opened`` /
    ``blockedByFoe``) finds no blocking foe. Vehicles still inside the junction
    after their phase ended are committed and clear first.
    """

    def __init__(self, zone_map: ZoneMap, params: NegotiationParams | None = None):
        self.zones = zone_map
        self.p = params or NegotiationParams()
        self.stats = {k: 0 for k in (
            "in_zone", "both_in_zone", "cannot_stop", "committed", "natural",
            "mutual", "stalled_foe", "diverge", "seam_follow", "deadlock_break",
            "minor_wait", "minor_gap", "forced", "group", "major_patience", "committed_yields", "claimed", "clearing", "weave", "box_rule")}
        self.last_decisions: list[PairDecision] = []
        self.forced_events: list[tuple[float, str]] = []

    # ------------------------------------------------------------ helpers
    def impatience(self, ag: Agent):
        p = self.p
        if ag.wait_time >= p.green_priority_after:
            return p.impatience_max
        base = p.impatience_base_moto if ag.motorbike else p.impatience_base_car
        tti = p.time_to_impatience_moto if ag.motorbike else p.time_to_impatience_car
        growth = ag.wait_time / tti if tti > 0 else 0.0
        group = min(p.group_cap, p.group_gain * max(0, ag.group_size - 1))
        return max(0.0, min(p.impatience_max, base + growth + group))

    def timegap(self, ag):
        return self.p.timegap_moto if ag.motorbike else self.p.timegap_car

    def follow_gap(self, ag):
        return self.p.follow_gap_moto if ag.motorbike else self.p.follow_gap_car

    def max_yield(self, ag):
        return self.p.max_yield_moto if ag.motorbike else self.p.max_yield_car

    @staticmethod
    def is_minor(ag: Agent):
        # Permissive right turns must yield at their merge conflict just as
        # left turns yield to opposing traffic. Signal permission alone does
        # not give the right turn priority over the through stream.
        return ag.arm_green and ag.turn in ("left", "right")

    @staticmethod
    def arrival(ag: Agent, z_in):
        return travel_time(z_in - ag.front, ag.speed, ag.accel, ag.desired)

    @staticmethod
    def leave(ag: Agent, z_out):
        # Conservative clearing time: half acceleration, plus body length.
        return travel_time(z_out - ag.back, ag.speed, 0.5 * ag.accel, ag.desired)

    @staticmethod
    def can_stop(ag: Agent, z_in, decel=None):
        decel = ag.max_decel if decel is None else decel
        room = z_in - ag.min_gap - ag.front
        return ag.speed * ag.speed / (2.0 * max(decel, 0.1)) <= max(0.0, room) + 0.05

    @staticmethod
    def clearing(ag: Agent):
        return ag.on_junction and not ag.arm_green

    @staticmethod
    def idle(ag: Agent):
        return ag.speed < 0.5 and (ag.stalled or ag.hold_time > 0.0)

    @staticmethod
    def in_zone(ag: Agent, z_in, z_out):
        return ag.front > z_in and ag.back < z_out

    # ------------------------------------------------------------ pair order
    def order_active(self, a: Agent, b: Agent, z: Zone):
        """Order two *active* vehicles (majors or committed) on zone ``z``."""
        a_in_z = self.in_zone(a, z.a_in, z.a_out)
        b_in_z = self.in_zone(b, z.b_in, z.b_out)
        if a_in_z != b_in_z:
            return (a, b, "in_zone") if a_in_z else (b, a, "in_zone")
        if a_in_z and b_in_z:
            pa = (a.front - z.a_in) / max(0.1, z.a_out - z.a_in + a.length)
            pb = (b.front - z.b_in) / max(0.1, z.b_out - z.b_in + b.length)
            return (a, b, "both_in_zone") if pa >= pb else (b, a, "both_in_zone")
        if self.p.idle_rule:
            idle_a, idle_b = self.idle(a), self.idle(b)
        else:
            idle_a, idle_b = a.stalled, b.stalled
        if idle_a != idle_b:
            # A stopped vehicle that is held (by its own leader or by another
            # conflict) will not pass now and must not hold this conflict
            # (SUMO: willPass == false / request withdrawn).
            return (b, a, "stalled_foe") if idle_a else (a, b, "stalled_foe")
        stop_a, stop_b = self.can_stop(a, z.a_in), self.can_stop(b, z.b_in)
        ta_in, tb_in = self.arrival(a, z.a_in), self.arrival(b, z.b_in)
        if stop_a != stop_b:
            return (b, a, "cannot_stop") if stop_a else (a, b, "cannot_stop")
        if not stop_a:
            return ((a, b, "cannot_stop") if (ta_in, a.vid) <= (tb_in, b.vid)
                    else (b, a, "cannot_stop"))
        ca, cb = a.commit_time is not None, b.commit_time is not None
        if ca and cb:
            return ((a, b, "committed") if (a.commit_time, -a.front, a.vid) <=
                    (b.commit_time, -b.front, b.vid) else (b, a, "committed"))
        if ca or cb:
            x, y = (a, b) if ca else (b, a)
            zx_in, zx_out = (z.a_in, z.a_out) if ca else (z.b_in, z.b_out)
            zy_in, zy_out = (z.b_in, z.b_out) if ca else (z.a_in, z.a_out)
            tx_in, tx_out = self.arrival(x, zx_in), self.leave(x, zx_out)
            ty_in, ty_out = self.arrival(y, zy_in), self.leave(y, zy_out)
            if ty_out + self.follow_gap(x) <= tx_in:
                return y, x, "natural"
            # The major yields to a committed vehicle if it can do so with a
            # firm but normal deceleration; otherwise the committed vehicle,
            # if it still can, lets the major pass (SUMO: brake for leader).
            if self.can_stop(y, zy_in, 1.5 * y.decel) or not self.can_stop(x, zx_in, 1.5 * x.decel):
                return x, y, "committed"
            return y, x, "committed_yields"
        # Do not let a continuous stream of equally green traffic keep
        # reordering ahead of a vehicle that has already waited several
        # seconds. This applies only after occupancy and braking checks above.
        threshold = self.p.green_priority_after
        aged_a, aged_b = a.wait_time >= threshold, b.wait_time >= threshold
        if aged_a != aged_b:
            return (a, b, "aged_priority") if aged_a else (b, a, "aged_priority")
        if aged_a and abs(a.wait_time - b.wait_time) >= 0.2:
            return ((a, b, "aged_priority") if a.wait_time > b.wait_time
                    else (b, a, "aged_priority"))
        return ((a, b, "mutual") if (ta_in, a.vid) <= (tb_in, b.vid)
                else (b, a, "mutual"))

    # ------------------------------------------------------------ minor gate
    def minor_blocked(self, m: Agent, foes, grouped: bool):
        """SUMO ``opened()`` for an uncommitted minor ``m``.

        ``foes`` is a list of ``(foe, zones)`` with zones seen from ``m``
        (z.a_* on m's path). Returns ``(blocker_vid | None, mode)`` where mode is
        "gap", "forced" or "group" when not blocked.
        """
        imp = self.impatience(m)
        mode = "gap"
        for f, zones in foes:
            for z in zones:
                if f.back >= z.b_out or m.back >= z.a_out:
                    continue
                if self.in_zone(f, z.b_in, z.b_out):
                    return f.vid, None
                tm_in, tm_out = self.arrival(m, z.a_in), self.leave(m, z.a_out)
                tf_in, tf_out = self.arrival(f, z.b_in), self.leave(f, z.b_out)
                if tf_out + self.follow_gap(m) <= tm_in:
                    continue  # pass behind the foe
                need = tm_out + self.timegap(m)
                if tf_in >= need:
                    continue  # pass ahead with the full time gap
                forceable = (not self.is_minor(f) and f.commit_time is None and
                             not f.stalled and self.can_stop(f, z.b_in))
                if not forceable:
                    if f.stalled and not self.in_zone(f, z.b_in, z.b_out):
                        continue
                    return f.vid, None
                if f.hold_time >= self.max_yield(f):
                    self.stats["major_patience"] += 1
                    return f.vid, None
                if grouped and self.can_stop(f, z.b_in, self.p.forced_decel_share * f.max_decel):
                    mode = "group"
                    continue
                if imp <= 0.0:
                    return f.vid, None
                dec = imp * self.p.forced_decel_share * f.max_decel
                t_brake = braking_arrival(z.b_in - f.min_gap - f.front, f.speed, dec)
                t_brake = tf_in + 30.0 if t_brake == INF else max(t_brake, tf_in)
                t_eff = (1.0 - imp) * tf_in + imp * t_brake
                if t_eff >= need and self.can_stop(f, z.b_in, dec):
                    if mode == "gap":
                        mode = "forced"
                    continue
                return f.vid, None
        return None, mode

    def weave(self, g: Agent, c: Agent, zones):
        """New-phase vehicle ``g`` against leftover ``c`` (zones seen from g).

        "behind": g arrives after c has cleared; "ahead": g clears every
        shared zone before c can arrive (or c is stuck behind its own leader);
        "wait": otherwise. A green vehicle never STOPS inside such a zone
        (enforced after all targets are known), so weaving cannot gridlock.
        """
        if not self.p.weave or any(self.in_zone(c, q.b_in, q.b_out) for q in zones):
            if self.p.weave:
                return "wait"
            behind_only = all(self.leave(c, q.b_out) + self.follow_gap(g) <= self.arrival(g, q.a_in)
                              for q in zones) and not any(self.in_zone(c, q.b_in, q.b_out) for q in zones)
            return "behind" if behind_only else "wait"
        tg = self.timegap(g)
        behind = ahead = True
        for q in zones:
            g_in, g_out = self.arrival(g, q.a_in), self.leave(g, q.a_out)
            c_in, c_out = self.arrival(c, q.b_in), self.leave(c, q.b_out)
            if not (c_out + self.follow_gap(g) <= g_in):
                behind = False
            if not (c.stalled or c_in >= g_out + tg):
                ahead = False
        if behind:
            return "behind"
        if ahead and (c.stalled or all(self.can_stop(c, q.b_in) for q in zones)):
            return "ahead"
        return "wait"

    def zone_verdict(self, m: Agent, f: Agent, z: Zone, leader, first_over):
        """May minor ``m`` enter zone ``z`` (z.a_* on m's path) shared with ``f``?

        Returns ("ok", mode) with mode "behind" (f passes first, no action),
        "ahead" (m clears before f arrives + timegap), "forced" (f can still
        brake: SUMO impatience), "group" (m follows a group member that is
        already going ahead of f), "claimed" (m is already in the zone), or
        ("block", reason).
        """
        if self.in_zone(m, z.a_in, z.a_out):
            return "ok", "claimed"
        if self.in_zone(f, z.b_in, z.b_out):
            if f.speed > 0.5:
                t_clear = self.leave(f, z.b_out)
                if self.arrival(m, z.a_in) >= t_clear + self.follow_gap(m):
                    return "ok", "behind"
            return "block", "in_zone"
        tm_in, tm_out = self.arrival(m, z.a_in), self.leave(m, z.a_out)
        tf_in, tf_out = self.arrival(f, z.b_in), self.leave(f, z.b_out)
        if tf_out + self.follow_gap(m) <= tm_in:
            return "ok", "behind"
        need = tm_out + self.timegap(m)
        if tf_in >= need:
            return "ok", "ahead"
        if f.on_junction and not f.arm_green:
            # Left over from the previous phase: weave through it only in a
            # clear gap (the box rule below keeps m from stopping in its way).
            mode = self.weave(m, f, [z])
            if mode == "ahead":
                return "ok", "ahead"
            if mode == "behind":
                return "ok", "behind"
            return "block", "clearing"
        if self.is_minor(f) or f.commit_time is not None:
            # Another left turner or a vehicle clearing the junction: whoever
            # reaches the zone first keeps it.
            if (tm_in, m.vid) < (tf_in, f.vid) and not self.in_zone(f, z.b_in, z.b_out):
                return "ok", "ahead"
            return "block", "mutual"
        if f.stalled:
            return "ok", "forced"  # stuck behind its own leader: not passing
        if tm_in >= tf_in:
            # SUMO applies impatience only when ego would reach the conflict
            # first (MSLink::blockedByFoe: arrivalTime < foe arrivalTime). A
            # through vehicle starting from the stop line reaches the conflict
            # in front of it before an opposing left turner does.
            return "block", "foe_first"
        if f.speed < 0.5 and f.hold_time > 0.0:
            # A foe waiting at the edge of this zone is pushed back only for a
            # while; then it inches in itself and reclaims its turn.
            reclaim = self.p.reclaim_after_moto if f.motorbike else self.p.reclaim_after_car
            if f.hold_time >= reclaim:
                self.stats["major_patience"] += 1
                return "block", "reclaim"
            return "ok", "forced"
        if not self.can_stop(f, z.b_in):
            return "block", "cannot_stop"
        if f.hold_time >= self.max_yield(f):
            self.stats["major_patience"] += 1
            return "block", "patience"
        if (leader is not None and tm_in < tf_in + 1.0 and
                any(g is leader for g in first_over.get(f.vid, ()))):
            if self.can_stop(f, z.b_in, self.p.forced_decel_share * f.max_decel):
                return "ok", "group"
        imp = self.impatience(m)
        if imp <= 0.0:
            return "block", "gap"
        dec = imp * self.p.forced_decel_share * f.max_decel
        t_brake = braking_arrival(z.b_in - f.min_gap - f.front, f.speed, dec)
        t_brake = tf_in + 30.0 if t_brake == INF else max(t_brake, tf_in)
        t_eff = (1.0 - imp) * tf_in + imp * t_brake
        if t_eff >= need and self.can_stop(f, z.b_in, dec):
            return "ok", "forced"
        return "block", "gap"

    # ------------------------------------------------------------ main step
    def decide(self, agents: Iterable[Agent], now=0.0):
        """Return ``(stop, commits)``.

        ``stop`` maps vid to ``(stop_s, obstacle_speed)`` (``inf`` = free);
        ``commits`` is the set of minors that commit to crossing this tick.
        """
        agents = list(agents)
        stop = {ag.vid: (INF, 0.0) for ag in agents}
        decisions = []
        edges = []  # [first, second, zones seen from second, reason]
        pending = [ag for ag in agents if not ag.passive and self.is_minor(ag)
                   and ag.commit_time is None]
        pending_ids = {ag.vid for ag in pending}
        zone_cache = {}

        def uncleared(a, b):
            k = (a.vid, b.vid)
            if k not in zone_cache:
                zone_cache[k] = [z for z in self.zones.zones(a.movement, b.movement)
                                 if a.back < z.a_out and b.back < z.b_out]
            return zone_cache[k]

        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                if a.corridor == b.corridor:
                    # Same connector: car following is done by the simulator
                    # on each road, but not across the stop-line seam.
                    if a.road != b.road and (a.movement == b.movement or
                                             self.zones.zones(a.movement, b.movement)):
                        lead, foll = (a, b) if a.front >= b.front else (b, a)
                        target = lead.back - foll.min_gap
                        if target < stop[foll.vid][0]:
                            stop[foll.vid] = (target, lead.speed)
                        self.stats["seam_follow"] += 1
                        decisions.append(PairDecision(lead.vid, foll.vid, None, "seam_follow"))
                    continue
                if a.passive or b.passive:
                    continue
                zones = uncleared(a, b)
                if not zones:
                    continue
                ca, cb = self.clearing(a), self.clearing(b)
                if ca != cb and a.arm != b.arm:
                    # Phase change: a vehicle left over from the previous phase
                    # crosses first; the new phase waits before (never inside)
                    # every zone it still has to cross, so no cross-phase
                    # gridlock can form in the middle of the junction.
                    c, g = (a, b) if ca else (b, a)
                    g_inside = any((self.in_zone(g, q.a_in, q.a_out) if g is a
                                    else self.in_zone(g, q.b_in, q.b_out)) for q in zones)
                    if not g_inside:
                        seen_g = [q if g is a else q.swapped() for q in zones]
                        mode = self.weave(g, c, seen_g)
                        if mode == "behind":
                            continue  # g reaches the zone after c has gone
                        if mode == "ahead":
                            # Lach doan: g slips through ahead of the leftover
                            # vehicle, which waits a moment if it must.
                            edges.append([g, c, [q.swapped() for q in seen_g], "weave"])
                            self.stats["weave"] += 1
                            continue
                        edges.append([c, g, seen_g, "clearing"])
                        continue
                z = min(zones, key=lambda q: min(q.a_in - a.front, q.b_in - b.front))
                if a.arm == b.arm and z.kind == "diverge":
                    # Same approach, paths that separate: plain car following
                    # behind the leader's back until the paths have split.
                    lead, foll = (a, b) if a.front >= b.front else (b, a)
                    target = lead.back - foll.min_gap
                    if target < stop[foll.vid][0]:
                        stop[foll.vid] = (target, lead.speed)
                    self.stats["diverge"] += 1
                    continue
                elif a.vid in pending_ids or b.vid in pending_ids:
                    # Majors ignore minors waiting at their wait point unless
                    # the minor's body is physically in the conflict zone.
                    m, f = (a, b) if a.vid in pending_ids else (b, a)
                    zm = z if m is a else z.swapped()
                    if self.in_zone(m, zm.a_in, zm.a_out) and not self.in_zone(f, zm.b_in, zm.b_out):
                        first, second, reason = m, f, "in_zone"
                    else:
                        continue  # handled by the minor gate below
                else:
                    first, second, reason = self.order_active(a, b, z)
                seen = [q if second is a else q.swapped() for q in zones]
                edges.append([first, second, seen, reason])

        self._break_cycles(edges)

        # Minor gate with creeping (nhich dan / cuop duong). A left turner does
        # not wait for one gap across the whole opposing stream: it enters the
        # junction and claims the conflict zones ONE BY ONE, in path order,
        # as each becomes takeable (gap, forcing a foe that can still brake,
        # or following its group). It stops just before the first zone it
        # cannot take, possibly in the middle of the stream, where through
        # vehicles of the sublanes it already blocks must stop or filter past.
        commits = set()
        active = [ag for ag in agents if not ag.passive and ag.vid not in pending_ids]
        first_over = {}   # foe vid -> minors that go ahead of it this tick
        claimed = []      # (minor, zones on its path it may enter this tick)
        for m in sorted(pending, key=lambda g: -g.front):
            items = []  # (zone on m's path, foe)
            for f in active + [c for c, _ in claimed]:
                if f is m or f.corridor == m.corridor or f.arm == m.arm:
                    continue
                for q in uncleared(m, f):
                    items.append((q, f))
            if not items:
                if m.on_junction:
                    commits.add(m.vid)
                    m.commit_time = now
                continue
            items.sort(key=lambda it: it[0].a_in)
            leader = None
            for g in pending:
                if (g is not m and g.arm == m.arm and g.turn == m.turn and
                        g.front > m.front and m.front >= g.back - self.p.group_gap):
                    leader = g
                    break
            target, taken, blocker = INF, [], None
            for q, f in items:
                if q.a_in >= target:
                    break
                verdict, mode = self.zone_verdict(m, f, q, leader, first_over)
                if verdict == "ok":
                    if mode in ("ahead", "forced", "group", "claimed"):
                        edges.append([m, f, [q.swapped()], mode if mode != "claimed" else "in_zone"])
                        first_over.setdefault(f.vid, []).append(m)
                        if mode in ("forced", "group"):
                            self.stats[mode] += 1
                            self.forced_events.append((now, m.vid))
                    taken.append(q)
                    continue
                if not self.can_stop(m, q.a_in) and (
                        self.can_stop(f, q.b_in) or
                        self.arrival(m, q.a_in) < self.arrival(f, q.b_in)):
                    # Too close to stop: it goes, the foe must yield if it can.
                    edges.append([m, f, [q.swapped()], "cannot_stop"])
                    first_over.setdefault(f.vid, []).append(m)
                    taken.append(q)
                    continue
                target = min(target, q.a_in - m.min_gap)
                blocker = blocker or f
            claimed.append((m, taken))
            if math.isfinite(target):
                self.stats["minor_wait"] += 1
                if target < stop[m.vid][0]:
                    stop[m.vid] = (target, 0.0)
                decisions.append(PairDecision(blocker.vid, m.vid, None, "minor_wait"))
            else:
                self.stats["minor_gap"] += 1

        for first, second, seen, reason in edges:
            target = INF
            for zs in seen:  # a_* on the second vehicle's path
                if first.back >= zs.b_out or second.back >= zs.a_out:
                    continue
                if reason not in ("in_zone", "both_in_zone", "diverge", "clearing"):
                    t_second = self.arrival(second, zs.a_in)
                    if t_second >= self.leave(first, zs.b_out) + self.follow_gap(second):
                        continue  # jmAdvance: cannot arrive before the zone is clear
                elif reason == "in_zone" and first.speed > 0.5:
                    t_second = self.arrival(second, zs.a_in)
                    if t_second >= self.leave(first, zs.b_out) + self.follow_gap(second):
                        continue
                target = min(target, zs.a_in - second.min_gap)
            if target < stop[second.vid][0]:
                stop[second.vid] = (target, 0.0)
            if math.isfinite(target):
                self.stats[reason] = self.stats.get(reason, 0) + 1
                decisions.append(PairDecision(first.vid, second.vid, seen[0], reason))
        # Box rule: a green vehicle never stops with its body inside a zone
        # that a vehicle left over from the previous phase still has to cross.
        leftovers = [c for c in agents if self.clearing(c) and not c.passive]
        if leftovers:
            for g in agents:
                if g.passive or self.clearing(g) or not math.isfinite(stop[g.vid][0]):
                    continue
                zs = []
                for c in leftovers:
                    if c.arm == g.arm:
                        continue
                    zs.extend(q for q in uncleared(g, c) if g.front <= q.a_in)
                if not zs:
                    continue
                target, spd = stop[g.vid]
                for q in sorted(zs, key=lambda q: -q.a_in):
                    if q.a_in - g.min_gap < target < q.a_out + g.length + g.min_gap:
                        target, spd = q.a_in - g.min_gap, 0.0
                        self.stats["box_rule"] += 1
                stop[g.vid] = (target, spd)
        self.last_decisions = decisions
        return stop, commits

    def _break_cycles(self, edges):
        """Flip one non-physical edge per wait-for cycle (longest-held goes)."""
        physical = ("in_zone", "both_in_zone", "diverge", "cannot_stop", "clearing")
        waits = {}
        for item in edges:
            waits.setdefault(item[1].vid, []).append(item)
        state, stack = {}, []

        def visit(vid):
            state[vid] = 1
            stack.append(vid)
            for item in list(waits.get(vid, ())):
                nxt = item[0].vid
                if state.get(nxt) == 1:
                    cycle = set(stack[stack.index(nxt):])
                    cands = [it for v in cycle for it in waits.get(v, ())
                             if it[0].vid in cycle and it[3] not in physical]
                    if cands:
                        e = max(cands, key=lambda it: (it[1].hold_time, it[1].front))
                        e[0], e[1] = e[1], e[0]
                        e[2] = [q.swapped() for q in e[2]]
                        e[3] = "deadlock_break"
                        self.stats["deadlock_break"] += 1
                elif state.get(nxt) is None:
                    visit(nxt)
            stack.pop()
            state[vid] = 2

        for vid in list(waits):
            if state.get(vid) is None:
                visit(vid)


# ---------------------------------------------------------------------------
# Adapter for python_sim.LeVanLuongSimulator (safe mode)
# ---------------------------------------------------------------------------

class SimulatorAdapter:
    """Builds agents from a ``LeVanLuongSimulator`` and publishes, per tick,

    * ``sim._junction_grants``: approach vehicles allowed to cross the stop line
      (they may enter and wait inside the junction, like SUMO internal junctions);
    * ``sim._junction_stop``: ``{creation_id: route coordinate}`` that the
      vehicle's *front* must not pass this tick.
    """

    def __init__(self, sim, params: NegotiationParams | None = None,
                 lateral_margin=0.2, sample_step=0.5, group_window=12.0):
        self.sim = sim
        self.group_window = float(group_window)
        self.sample_step = float(sample_step)
        if isinstance(params, dict):
            params = NegotiationParams(**params)
        self.zone_map = ZoneMap(self._sample, lateral_margin=lateral_margin)
        self.negotiator = JunctionNegotiator(self.zone_map, params)
        self.wait: dict[int, float] = {}
        self.hold: dict[int, float] = {}
        self.commit: dict[int, float] = {}
        self._seam_off: dict[int, float] = {}
        self._conflict_held: set[int] = set()
        self.hard_caps = 0
        self.creep_min = 0.2
        self.sweep_car = 0.75
        self.sweep_moto = 0.0
        self.box_clear = False
        self.filter_enabled = True
        self.filter_cooldown = 1.0
        self.filter_moves = 0

    def _sample(self, movement_id):
        if "#" in movement_id:
            # Continuous lateral band: "<connector>#<centre shift>:<half width>".
            base, spec = movement_id.split("#", 1)
            shift, half_width = (float(x) for x in spec.split(":"))
        else:
            base, spec = movement_id.split("@", 1)
            lane, occupied = (int(x) for x in spec.split(":"))
            shift = (lane + (occupied - 1) / 2.0) * 1.05
            half_width = 0.85 if occupied == 2 else 0.35
        road = self.sim._geometry_roads[base]
        n = max(2, int(math.ceil(road.length / self.sample_step)) + 1)
        s = np.linspace(0.0, road.length, n)
        pts = np.array([(p.x, p.y) for p in (road.pose(float(x), shift) for x in s)])
        return PathSample(pts, s, half_width, road.length)

    def _band_path(self, v, base, on_conn):
        """Lateral band the vehicle will actually sweep on its connector.

        Lane indices are not enough: crossing the stop-line seam can rebase the
        lateral shift and the vehicle then drifts to its target sublane, so the
        sublane path of its lane index may be metres away from its body.
        """
        sim = self.sim
        # Body half width plus the sweep of a rectangle on a curved path
        # (front/rear corners leave the centre-line band; SUMO adds a sagitta).
        half = (0.85 + self.sweep_car) if v.occupied == 2 else (0.35 + self.sweep_moto)
        if on_conn:
            lo = min(v.current_lateral_shift, v.target_lateral_shift)
            hi = max(v.current_lateral_shift, v.target_lateral_shift)
        else:
            shift = sim._continuous_seam_pose(f"{v.arm}_in", base, v.current_lateral_shift)[1]
            lane = sim._mapped_lane(v, v.lane, f"{v.arm}_in", base)
            target = (lane + (v.occupied - 1) / 2.0) * 1.05
            n_lanes = sim.roads[base]["num_lanes"]
            min_shift = (v.occupied - 1) / 2.0 * 1.05
            max_shift = (n_lanes - 1 - (v.occupied - 1) / 2.0) * 1.05
            if min_shift - 0.05 <= shift <= max_shift + 0.05:
                lo = hi = shift
            else:
                lo, hi = min(shift, target), max(shift, target)
        q = 0.15
        centre = round((lo + hi) / 2.0 / q) * q
        extra = math.ceil(((hi - lo) / 2.0 + abs((lo + hi) / 2.0 - centre)) / 0.1) * 0.1
        return f"{base}#{centre:.2f}:{half + extra:.2f}"

    def step(self, dt):
        sim = self.sim
        now = sim.cycle * dt
        agents, by_vid = [], {}
        # Clearing a stale entry token mirrors the reservation mode.
        for v in sim.vehicles:
            if (v.junction_entry_token and v.road == f"{v.arm}_in"
                    and v.position <= sim._stop_position(v) + 1e-8
                    and not (sim._green_for(v) if hasattr(sim, "_green_for") else sim._arm_is_green(v.arm))):
                v.junction_entry_token = False
        live = set()
        for v in sim.vehicles:
            road, _ = sim._road_state(v)
            base = f"{v.arm}_{v.to_arm}"
            on_conn = road == base
            on_in = road == f"{v.arm}_in"
            if not (on_conn or on_in):
                continue
            vid = v.creation_id
            live.add(vid)
            w = self.wait.get(vid, 0.0)
            self.wait[vid] = w + dt if v.speed < 1.0 else max(0.0, w - 2 * dt)
            held = vid in self._conflict_held and v.speed < 1.0
            self.hold[vid] = self.hold.get(vid, 0.0) + dt if held else 0.0
            green = (sim._green_for(v) if hasattr(sim, "_green_for") else sim._arm_is_green(v.arm))
            if on_in:
                stop_pos = sim._stop_position(v)
                committed = v.position > stop_pos + 1e-8 or v.junction_entry_token
                horizon = (max(30.0, v.speed * v.speed / (2.0 * max(1.0, v.decel)) + v.speed * 0.8 + 6.0)
                           if v.speed > 3.0 else max(10.0, v.speed * 2.0 + 4.0))
                passive = not committed and not green
                if passive and stop_pos - v.position > 8.0:
                    continue
                if not committed and green and stop_pos - v.position > horizon:
                    continue
            else:
                passive = False
            movement = self._band_path(v, base, on_conn)
            # Connector progress of the front. Crossing the stop line rebases
            # progress through the seam pose, so an approach vehicle's
            # distance past the line is shifted by its seam offset.
            seam_off = (0.0 if on_conn else sim._continuous_seam_pose(
                f"{v.arm}_in", base, v.current_lateral_shift)[0])
            self._seam_off[vid] = seam_off
            front = v.position + v.length / 2.0 - v.approach_end + seam_off
            ag = Agent(vid=str(vid), movement=movement, arm=v.arm, turn=v.maneuver,
                       front=front, length=v.length, speed=max(0.0, v.speed),
                       accel=v.accel, decel=v.decel,
                       max_decel=min(7.5, 2.0 * v.decel), min_gap=max(0.5, v.gap),
                       desired=min(v.desired, 40.0 / 3.6), motorbike=v.occupied == 1,
                       on_junction=on_conn or v.position + v.length / 2.0 > v.approach_end,
                       arm_green=green,
                       wait_time=self.wait[vid], road=road, hold_time=self.hold[vid],
                       commit_time=self.commit.get(vid), passive=passive,
                       stalled=(v.speed < 0.5 and v.leading_distance < max(0.5, v.gap) + 1.5))
            agents.append(ag)
            by_vid[ag.vid] = v
        for vid in tuple(self.wait):
            if vid not in live:
                del self.wait[vid]
                self.hold.pop(vid, None)
                self.commit.pop(vid, None)
        # Group size: same-approach, same-movement vehicles bunched just behind.
        groups = {}
        for ag in agents:
            if ag.turn == "through":
                continue
            groups.setdefault((ag.arm, ag.corridor), []).append(ag)
        for members in groups.values():
            members.sort(key=lambda g: -g.front)
            for i, ag in enumerate(members):
                ag.group_size = 1 + sum(
                    1 for other in members[i + 1:]
                    if ag.front - other.front <= self.group_window and other.speed < 4.0)
        for ag in agents:
            vid = int(ag.vid)
            if ag.on_junction and not ag.arm_green and vid not in self.commit:
                self.commit[vid] = now  # clearing after the phase ended
                ag.commit_time = now
        stop, commits = self.negotiator.decide(agents, now)
        # A seam follower is longitudinal traffic, not a crossing foe. The
        # negotiator's fixed seam stop can appear only after its leader enters
        # the connector, forcing an impossible one-tick stop. Follow the
        # leader's moving rear bumper instead, including its next-tick speed.
        for decision in self.negotiator.last_decisions:
            if decision.reason != "seam_follow":
                continue
            leader = by_vid.get(decision.first)
            follower = by_vid.get(decision.second)
            if (leader is None or follower is None or
                    leader.arm != follower.arm or leader.to_arm != follower.to_arm or
                    leader.position <= follower.position):
                continue
            stopping_front = (leader.position - leader.length / 2.0 -
                              follower.gap)
            stop[decision.second] = (stopping_front - follower.approach_end +
                                     self._seam_off.get(follower.creation_id, 0.0),
                                     leader.speed)
        for vid in commits:
            self.commit.setdefault(int(vid), now)
        sim._junction_stop = {}
        for vid in tuple(self._seam_off):
            if vid not in live:
                del self._seam_off[vid]
        grants = set()
        # Keep the box clear for clearing traffic (SUMO: a vehicle may not
        # enter while a foe that is already on the junction is in its way).
        # A vehicle whose arm has green waits at the stop line as long as a
        # vehicle left over from the previous phase still has to cross its path.
        clearing = ([ag for ag in agents if ag.on_junction and not ag.arm_green]
                    if self.box_clear else [])
        blocked_by_clearing = set()
        if clearing:
            for ag in agents:
                if ag.on_junction or ag.passive or not ag.arm_green:
                    continue
                for c in clearing:
                    if c.arm != ag.arm and any(
                            ag.back < z.a_out and c.back < z.b_out
                            for z in self.zone_map.zones(ag.movement, c.movement)):
                        blocked_by_clearing.add(ag.vid)
                        break
        storage = {}
        for ag in agents:
            if ag.on_junction and ag.turn == "left" and ag.commit_time is None:
                storage[ag.arm] = storage.get(ag.arm, 0) + (1 if ag.motorbike else 2)
        for ag in agents:
            v = by_vid[ag.vid]
            s, obstacle_speed = stop[ag.vid]
            if math.isfinite(s):
                sim._junction_stop[v.creation_id] = (
                    v.approach_end + s - self._seam_off.get(v.creation_id, 0.0), obstacle_speed)
            if ag.on_junction or v.road != f"{v.arm}_in":
                continue
            committed = (v.position > sim._stop_position(v) + 1e-8) or v.junction_entry_token
            # An earlier grant cannot be revoked only metres from the stop
            # line after the vehicle has passed its braking point. The
            # connector provides storage while it yields at the exit merge.
            dilemma_hold = (v.creation_id in sim._junction_grants and
                            v.speed >= 4.0 and
                            sim._stop_position(v) - v.position <=
                            v.speed ** 2 / (2.0 * max(1.0, v.decel)))
            room_ok = ((committed and v.maneuver != "right") or
                       dilemma_hold or sim._destination_has_room(v))
            # Enter only if the whole body fits before the wait point, and
            # (for left turns) while waiting storage inside the junction is left.
            store_ok = (ag.turn != "left" or not math.isfinite(s) or
                        storage.get(ag.arm, 0) < self.negotiator.p.wait_storage)
            # Creeping: a vehicle may cross the stop line whenever it can
            # move forward at all; it then inches up to its stop point
            # inside the junction instead of waiting behind the line.
            if ((s > self.creep_min or not math.isfinite(s)) and room_ok and store_ok
                    and not ag.passive and ag.vid not in blocked_by_clearing):
                grants.add(v.creation_id)
                v.junction_wait_since = None
                if v.position + max(0.0, v.speed) * dt >= sim._stop_position(v):
                    v.junction_entry_token = True
            elif v.junction_wait_since is None:
                v.junction_wait_since = now
        sim._junction_grants = grants
        self._last_blocked_by_clearing = blocked_by_clearing
        self._conflict_held = {int(d.second) for d in self.negotiator.last_decisions
                               if d.reason not in ("seam_follow", "diverge")}
        if self.filter_enabled:
            self._filter(agents, by_vid, stop, dt)
        return stop

    # ------------------------------------------------------------ filtering
    def _filter(self, agents, by_vid, stop, dt):
        """Lach: a motorbike held inside the junction by a crossing foe shifts
        one or two sublanes sideways when that lateral band is free and does
        not run into the foe that holds it (cf. SUMO SL2015 sublane changes,
        lateral gap selection of Lee, Polak & Bell 2009)."""
        sim = self.sim
        blockers = {}
        for d in self.negotiator.last_decisions:
            if d.reason not in ("seam_follow", "diverge"):
                blockers.setdefault(d.second, []).append(d.first)
        ag_by_vid = {ag.vid: ag for ag in agents}
        for ag in agents:
            # A permissive right turn merges with a through stream. A lateral
            # filtering move changes its conflict band after the through
            # vehicle has committed, so keep it in its curb corridor here.
            if (not ag.motorbike or ag.turn != "left" or
                    not ag.on_junction or ag.vid not in blockers):
                continue
            v = by_vid[ag.vid]
            road = f"{v.arm}_{v.to_arm}"
            if v.road != road or v.lane_change_progress < 1.0:
                continue
            if v.time_since_lane_change < self.filter_cooldown:
                continue
            s_stop, _ = stop[ag.vid]
            if not math.isfinite(s_stop) or s_stop - ag.front > 6.0:
                continue  # not (yet) held close to a conflict
            foes = [ag_by_vid[f] for f in blockers[ag.vid] if f in ag_by_vid]
            if not foes:
                continue
            n_lanes = sim.roads[road]["num_lanes"]
            prog = v.position - v.approach_end
            best = None
            for dl in (1, -1, 2, -2):
                lane = v.lane + dl
                if not 0 <= lane <= n_lanes - v.occupied:
                    continue
                target = (lane + (v.occupied - 1) / 2.0) * 1.05
                lo, hi = min(v.current_lateral_shift, target), max(v.current_lateral_shift, target)
                # physical room on the connector beside / ahead of the bike
                room = True
                for o in sim.vehicles:
                    if o is v or o.road != road:
                        continue
                    op = o.position - o.approach_end
                    if abs(op - prog) - (o.length + v.length) / 2 >= 1.0:
                        continue
                    olo = min(o.current_lateral_shift, o.target_lateral_shift)
                    ohi = max(o.current_lateral_shift, o.target_lateral_shift)
                    if max(lo - ohi, olo - hi, 0.0) < 0.35 + (0.85 if o.occupied == 2 else 0.35) + 0.1:
                        room = False
                        break
                if not room:
                    continue
                q = 0.15
                centre = round(target / q) * q
                cand = f"{road}#{centre:.2f}:{0.35 + self.sweep_moto + math.ceil(abs(target - centre) / 0.1) * 0.1:.2f}"
                # the new band must get past every foe that holds the bike now
                clear = True
                for f in foes:
                    for z in self.zone_map.zones(cand, f.movement):
                        if z.a_out > ag.back and f.back < z.b_out:
                            clear = False
                            break
                    if not clear:
                        break
                if clear:
                    # nothing else may cross the new band right where the
                    # bike is or within a few metres ahead of it
                    for g in agents:
                        if g is ag or g.corridor == ag.corridor or g.passive:
                            continue
                        for z in self.zone_map.zones(cand, g.movement):
                            if (g.back < z.b_out and z.a_out > ag.back - 0.5 and
                                    z.a_in < ag.front + 4.0):
                                clear = False
                                break
                        if not clear:
                            break
                if clear:
                    best = lane
                    break
            if best is None:
                continue
            v.lane_change_from = v.lane
            v.lane = best
            v.start_lateral_shift = v.current_lateral_shift
            v.target_lateral_shift = (best + (v.occupied - 1) / 2.0) * 1.05
            v.lane_change_progress = 0.0
            v.time_since_lane_change = 0.0
            dist = abs(v.target_lateral_shift - v.start_lateral_shift)
            v.lateral_duration = max(0.4, math.sqrt(6.0 * max(0.4, dist) / 2.0) * 0.6)
            self.filter_moves += 1
