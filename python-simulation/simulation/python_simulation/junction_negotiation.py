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

from dataclasses import dataclass, field, replace
import math
import zlib
from typing import Callable, Iterable

import numpy as np

TURN_RANK = {"through": 0, "right": 1, "left": 2}
# movement id -> corridor (pure function of the string; read ~10^4 times a tick)
_CORRIDORS: dict[str, str] = {}
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
        self._oriented: dict[tuple[str, str], tuple[Zone, ...]] = {}
        self._samples: dict[str, PathSample] = {}

    def sample(self, movement):
        if movement not in self._samples:
            self._samples[movement] = self.sampler(movement)
        return self._samples[movement]

    def zones(self, a: str, b: str) -> tuple[Zone, ...]:
        # Both orientations are memoised: the swapped tuple used to be
        # rebuilt on every call (hundreds of thousands per 1,000 ticks).
        cached = self._oriented.get((a, b))
        if cached is not None:
            return cached
        if a == b:
            return ()
        key = (a, b) if a < b else (b, a)
        if key not in self._cache:
            self._cache[key] = self._compute(*key)
        zones = self._cache[key]
        zones = zones if key[0] == a else tuple(z.swapped() for z in zones)
        self._oriented[(a, b)] = zones
        return zones

    def _compute(self, a, b):
        pa, pb = self.sample(a), self.sample(b)
        thr = pa.half_width + pb.half_width + self.lateral_margin
        lo_a, hi_a = pa.points.min(0) - thr, pa.points.max(0) + thr
        lo_b, hi_b = pb.points.min(0), pb.points.max(0)
        if (hi_a < lo_b).any() or (hi_b < lo_a).any():
            return ()
        dx = pa.points[:, None, 0] - pb.points[None, :, 0]
        dy = pa.points[:, None, 1] - pb.points[None, :, 1]
        # Add the two squared coordinates directly. For integer samples,
        # preserve the accumulator dtype NumPy uses for sum().
        sum_dtype = dx[:0].sum().dtype if dx.dtype.kind in "iu" else dx.dtype
        close = np.add(dx * dx, dy * dy, dtype=sum_dtype) < thr * thr
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
    # Motorcycles in its left-turn group (Vu & Shimizu 2010 grouping:
    # members within 4 m of the next member, same movement), itself included.
    group_moto: int = 1
    road: str = ""
    hold_time: float = 0.0  # time held by a junction stop target at low speed
    commit_time: float | None = None  # passed its wait point (SUMO conflict entry)
    passive: bool = False  # red-arm approach vehicle: only seam following
    stalled: bool = False  # stopped behind its own leader: "will not pass" (SUMO willPass)
    lateral_shift: float = 0.0
    half_width: float = 0.35
    connector_shift: float = 0.0
    # Lateral interval (connector coordinates) the body centre sweeps: its
    # shift now and the shift it is moving to. None: use ``connector_shift``.
    connector_span: tuple | None = None
    # Pose in the plane (centre and unit heading); None when unknown.
    xy: tuple | None = None
    heading: tuple | None = None
    # Points along the axis of a long rigid body (rear quarter, centre,
    # front quarter) with its heading, ((x, y, tx, ty), ...); None for cars
    # and motorbikes.
    units: tuple | None = None

    @property
    def back(self):
        return self.front - self.length

    @property
    def corridor(self):
        movement = self.movement
        corridor = _CORRIDORS.get(movement)
        if corridor is None:
            corridor = _CORRIDORS[movement] = movement.replace("#", "@").split("@", 1)[0]
        return corridor


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
    # Longitudinal bunching (m) for a follower to join a crossing group:
    # Vu & Shimizu (2010) l0 = 4 m to the nearest member.
    group_gap: float = 4.0
    # Storage for minors waiting inside the junction, per approach, in
    # motorbike equivalents (car = 2). Further minors wait at the stop line.
    wait_storage: int = 10**6
    # Treat stopped+held vehicles as not passing (True) or only those stopped
    # behind their own leader (False).
    idle_rule: bool = False
    # New-phase vehicles may slip through gaps among leftovers (lach doan).
    weave: bool = True
    # Keep the box clear: a vehicle whose arm has just turned green waits at
    # the stop line while a vehicle left over from the previous phase still
    # has to cross its path. Without it, short phases (an exploring RL agent
    # switching every decision) fill the box from both directions and the
    # junction gridlocks permanently (14/11 audit, section 2.3).
    box_clear: bool = True
    # A wait cycle made only of bodies standing in each other's conflict
    # zones has no ordering left to change. After this many seconds the
    # longest-held vehicle is released to squeeze through (nhich qua): it
    # creeps forward under the body-overlap guard only.
    squeeze_after: float = 8.0
    # Gap acceptance of left turners against a green through stream.
    # "logit": binary logit of Vu & Shimizu (2010), Table 1, separate for
    # motorcycle and car leaders, one random draw per (left turner, through
    # vehicle) gap event. "sumo": deterministic time gap + impatience (the
    # model before 16/11/2026).
    gap_model: str = "logit"
    # V = b_tg * TG + b_ewt * EWT + b_mc * MC1 + b_v1 * V1 + const
    logit_moto: tuple = (2.449, -0.045, 0.198, 0.0, -2.257)
    logit_car: tuple = (2.887, 0.127, 0.0, 0.165, -4.225)
    # A gap that makes the through vehicle brake is taken only if it can stop
    # before the zone at this share of its emergency deceleration.
    logit_force_decel_share: float = 0.6
    # Before the decision point (comfortable stopping distance + margin) a
    # left turner only goes on when every driver would accept the gap.
    logit_far_gap: float = 3.0
    decision_margin: float = 2.0
    # Lateral evasion of a through vehicle facing a crossing left turner:
    # "cut tail" (pass behind it) or "run in front" (pass ahead of it) by a
    # sideways move inside the junction. Propensity per gap event of a
    # straight-go group leader (Vu & Shimizu 2010, Fig. 6: motorcycle leaders
    # 20 % cut tail + 5 % run in front, car leaders 5 % + 2 %). The direction
    # is the target band needing the smallest deviation that lets the
    # vehicle go on without stopping (Trinh et al. 2018, Eq. 3). Heading
    # limits: motorcycles may steer within the +-30 deg direction fan of
    # Trinh et al. (2018); cars keep the +-0.2 rad cone of Kanagaraj &
    # Treiber (2018).
    evade: bool = True
    # Willingness per gap event (independent draws). The realised shares
    # are lower: the manoeuvre must also be possible (room, timing, heading
    # cone). Calibrated so the realised shares match Vu & Shimizu (2010).
    evade_cut_moto: float = 1.0
    evade_front_moto: float = 0.18
    evade_cut_car: float = 0.65
    evade_front_car: float = 0.12
    # Heavy vehicles (trucks, buses) hold their path: the observations above
    # cover motorcycle and car leaders only.
    evade_cut_heavy: float = 0.0
    evade_front_heavy: float = 0.0
    # Run in front forces the left turner to give way: only if it has not
    # committed yet and can stop with this share of its maximum braking.
    evade_force_decel_share: float = 0.5
    lateral_cone_moto: float = 0.5236
    lateral_cone_car: float = 0.2
    evade_max_sublanes: int = 3
    evade_min_speed: float = 1.0
    evade_leader_gap: float = 4.0
    evade_pet: float = 0.5


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
    claim the conflict zones of their path one by one, in path order, when
    :meth:`zone_verdict` (SUMO ``opened`` / ``blockedByFoe``) finds the zone
    takeable, and wait before the first zone they cannot take. Vehicles still inside the junction
    after their phase ended are committed and clear first.
    """

    def __init__(self, zone_map: ZoneMap, params: NegotiationParams | None = None):
        self.zones = zone_map
        self.p = params or NegotiationParams()
        self.stats = {k: 0 for k in (
            "in_zone", "both_in_zone", "cannot_stop", "committed", "natural",
            "mutual", "stalled_foe", "diverge", "seam_follow", "deadlock_break",
            "minor_wait", "minor_gap", "forced", "group", "major_patience", "committed_yields", "claimed", "clearing", "weave", "box_rule", "conceded",
            "cut_tail", "run_in_front", "swerve")}
        self.last_decisions: list[PairDecision] = []
        self.forced_events: list[tuple[float, str]] = []
        self._hard_held: dict[str, list] = {}
        # Random stream id (the simulator seed) and gap events already accepted.
        self.seed = 0
        self._accepted: set[tuple[str, str]] = set()
        self._seen_gaps: set[tuple[str, str]] = set()
        # (minor vid, major vid): the minor gave way to a major that ran in
        # front of it with an evasive move (Vu & Shimizu strategy B1).
        self._conceded: set[tuple[str, str]] = set()
        # (left turner vid, through vid, TG, P(accept), accepted) per decision.
        self.gap_events: list[tuple] = []

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
    def laterally_apart(a: Agent, b: Agent, margin=0.15):
        """Bodies on the same connector (or about to be) that do not share a
        lateral band, now or while finishing a lateral move."""
        a_lo, a_hi = a.connector_span or (a.connector_shift, a.connector_shift)
        b_lo, b_hi = b.connector_span or (b.connector_shift, b.connector_shift)
        return max(a_lo - b_hi, b_lo - a_hi) >= a.half_width + b.half_width + margin

    @staticmethod
    def apart_in_plane(a: Agent, b: Agent, margin=0.15):
        """Sideways distance between the two centres, across the heading of
        the rear vehicle, exceeds both half widths.

        A long rigid body is tested at several points of its axis: a turning
        truck whose centre is already well to the side can still have its
        rear right beside a vehicle that follows it out of the same approach.
        """
        if a.xy is None or b.xy is None or a.heading is None or b.heading is None:
            return False
        units_a = a.units or ((a.xy[0], a.xy[1], a.heading[0], a.heading[1]),)
        units_b = b.units or ((b.xy[0], b.xy[1], b.heading[0], b.heading[1]),)
        need = a.half_width + b.half_width + margin
        for ax, ay, atx, aty in units_a:
            for bx, by, btx, bty in units_b:
                dx, dy = bx - ax, by - ay
                rear = (atx, aty) if dx * atx + dy * aty >= 0.0 else (btx, bty)
                if abs(dx * rear[1] - dy * rear[0]) < need:
                    return False
        return True

    @staticmethod
    def clearing(ag: Agent):
        return ag.on_junction and not ag.arm_green

    @staticmethod
    def idle(ag: Agent):
        return ag.speed < 0.5 and (ag.stalled or ag.hold_time > 0.0)

    @staticmethod
    def in_zone(ag: Agent, z_in, z_out):
        return ag.front > z_in and ag.back < z_out

    # ------------------------------------------------------------ gap acceptance
    def total_gap(self, m: Agent, f: Agent, z: Zone):
        """Vu & Shimizu (2010) total gap = longitudinal gap - lateral gap (s).

        Longitudinal gap: time the through vehicle needs to reach the conflict
        (its distance over its speed; accelerating from rest when slow).
        Lateral gap: time the left turner needs to cross and clear it.
        """
        dist = z.b_in - f.front
        if f.speed >= 2.0:
            longitudinal = max(0.0, dist) / f.speed
        else:
            longitudinal = self.arrival(f, z.b_in)
        return longitudinal - self.leave(m, z.a_out)

    def accept_probability(self, m: Agent, tg: float):
        b_tg, b_ewt, b_mc, b_v1, const = (self.p.logit_moto if m.motorbike
                                          else self.p.logit_car)
        v = (b_tg * tg + b_ewt * m.wait_time + b_mc * m.group_moto +
             b_v1 * m.speed + const)
        v = max(-60.0, min(60.0, v))
        return 1.0 / (1.0 + math.exp(-v))

    def _draw(self, a: str, b: str):
        """Uniform draw fixed per (driver, gap): independent of tick and of
        the process hash seed."""
        return zlib.crc32(f"{self.seed}:{a}:{b}".encode()) / 2.0 ** 32

    def logit_verdict(self, m: Agent, f: Agent, z: Zone, leader, first_over,
                      tm_in, tm_out, tf_in):
        """Left turner ``m`` against green through vehicle ``f`` (Vu & Shimizu)."""
        if f.stalled or (f.speed < 0.5 and
                         any(first is not m for first in self._hard_held.get(f.vid, ()))):
            return "ok", "forced"  # f is not coming now
        if f.speed < 0.5 and f.hold_time > 0.0:
            reclaim = self.p.reclaim_after_moto if f.motorbike else self.p.reclaim_after_car
            if f.hold_time >= reclaim:
                self.stats["major_patience"] += 1
                return "block", "reclaim"
        clear_ahead = tf_in >= tm_out + self.timegap(m)
        can_brake = self.can_stop(f, z.b_in, self.p.logit_force_decel_share * f.max_decel)
        if (leader is not None and any(g is leader for g in first_over.get(f.vid, ()))
                and (clear_ahead or can_brake)):
            # Assumption 1 of Vu & Shimizu: the group follows its leader.
            return "ok", "group"
        tg = self.total_gap(m, f, z)
        key = (m.vid, f.vid)
        if key not in self._accepted:
            decision_dist = (m.speed * m.speed / (2.0 * max(m.decel, 0.1)) +
                             m.min_gap + self.p.decision_margin)
            if z.a_in - m.front > decision_dist:
                # Not yet at the decision point.
                if tg >= self.p.logit_far_gap and clear_ahead:
                    return "ok", "ahead"
                return "block", "approach"
            p = self.accept_probability(m, tg)
            accepted = self._draw(m.vid, f.vid) < p
            if key not in self._seen_gaps:
                # One record per gap event (the first decision), as observed
                # in the field; later re-evaluations use the same draw.
                self._seen_gaps.add(key)
                self.gap_events.append((m.vid, f.vid, tg, p, accepted, m.motorbike))
            if not accepted:
                return "block", "gap"
            self._accepted.add(key)
        if clear_ahead:
            return "ok", "ahead"
        if can_brake:
            return "ok", "forced"
        return "block", "cannot_stop"

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
        """SUMO ``opened()`` for an uncommitted minor ``m`` (whole crossing).

        Not used by :meth:`decide`, which claims zones one by one through
        :meth:`zone_verdict`; kept as the reference single-wait-point gate.

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
            # The leftover is already in a shared zone (or weaving is off):
            # g can only go behind it. It does so whenever the leftover is
            # moving and will have left every shared zone before g gets
            # there. Being inside a zone used to mean "wait" outright, which
            # held a vehicle at the stop line for a leftover that was about
            # to leave a zone 30-40 m ahead of it.
            if self.idle(c) or c.speed < 0.5:
                return "wait"
            behind_only = all(self.leave(c, q.b_out) + self.follow_gap(g) <= self.arrival(g, q.a_in)
                              for q in zones)
            return "behind" if behind_only else "wait"
        tg = self.timegap(g)
        # A leftover that has been standing, held at a conflict, for a while
        # is not coming: waiting for it (a right turner parked in the merge
        # it must leave, yielding to a car that is itself blocked) closes a
        # ring that nothing else opens.
        stuck = c.stalled or (c.speed < 0.5 and c.hold_time >= self.p.squeeze_after)
        behind = ahead = True
        for q in zones:
            g_in, g_out = self.arrival(g, q.a_in), self.leave(g, q.a_out)
            c_in, c_out = self.arrival(c, q.b_in), self.leave(c, q.b_out)
            if not (c_out + self.follow_gap(g) <= g_in):
                behind = False
            if not (stuck or c_in >= g_out + tg):
                ahead = False
        if behind:
            return "behind"
        if ahead and (stuck or all(self.can_stop(c, q.b_in) for q in zones)):
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
        if (m.vid, f.vid) in self._conceded and not self.in_zone(m, z.a_in, z.a_out):
            # f runs in front of m: m brakes for it like for any major.
            return "block", "conceded"
        if (self.p.gap_model == "logit" and m.turn == "left" and f.arm_green and
                not self.is_minor(f) and f.commit_time is None):
            return self.logit_verdict(m, f, z, leader, first_over, tm_in, tm_out, tf_in)
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
        if f.speed < 0.5 and any(first is not m for first in self._hard_held.get(f.vid, ())):
            # Standing because a third vehicle physically blocks it (a body in
            # its zone, or a leftover it must let clear): like a foe stuck
            # behind its own leader, it will not pass now. Waiting for it
            # (or letting it "reclaim" its turn) closed rings such as
            # leftover -> left turner -> through -> leftover (seed 2, random
            # policy). If it is released later, m is in its zone first and
            # the in-zone order makes it wait.
            return "ok", "forced"
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
        live = {ag.vid for ag in agents}
        self._accepted = {k for k in self._accepted if k[0] in live and k[1] in live}
        self._seen_gaps = {k for k in self._seen_gaps if k[0] in live and k[1] in live}
        self._conceded = {k for k in self._conceded if k[0] in live and k[1] in live}
        decisions = []
        edges = []  # [first, second, zones seen from second, reason]
        pending = [ag for ag in agents if not ag.passive and self.is_minor(ag)
                   and ag.commit_time is None]
        pending_ids = {ag.vid for ag in pending}
        zone_cache = {}

        # Green vehicles that must yield to a leftover of the previous phase
        # on real timing (not merely share a zone with one), each with the
        # speed of the slowest leftover it yields to.
        self.clearing_holds = {}

        def uncleared(a, b):
            k = (a.vid, b.vid)
            if k not in zone_cache:
                zone_cache[k] = [z for z in self.zones.zones(a.movement, b.movement)
                                 if a.back < z.a_out and b.back < z.b_out]
            return zone_cache[k]

        zone_lookup = self.zones.zones
        oriented_get = self.zones._oriented.get
        # (corridor, passive, movement) per agent, read once per tick.
        keys = [(ag.corridor, ag.passive, ag.movement) for ag in agents]
        n_agents = len(agents)
        for i, a in enumerate(agents):
            a_corridor, a_passive, a_movement = keys[i]
            for j in range(i + 1, n_agents):
                b_corridor, b_passive, b_movement = keys[j]
                if a_corridor != b_corridor:
                    if a_passive or b_passive:
                        continue
                    # Paths that never meet: nothing to decide (as before).
                    met = oriented_get((a_movement, b_movement))
                    if met is None:
                        met = zone_lookup(a_movement, b_movement)
                    if not met:
                        continue
                b = agents[j]
                if a_corridor == b_corridor:
                    # Same connector: car following is done by the simulator
                    # on each road, but not across the stop-line seam.
                    if a.road != b.road and (a.movement == b.movement or
                                             self.zones.zones(a.movement, b.movement)):
                        # The zone map compares all points along two paths.
                        # Parallel motorcycle sublanes can therefore form a
                        # spurious all-length zone although their live bodies
                        # are laterally separated at the seam.
                        # This holds for every vehicle type: the conflict
                        # band of a car is widened for the sweep of its
                        # corners, so a car at the line "followed" whatever
                        # stood or moved one sublane to its side and was kept
                        # there with a clear lane in front of it. On the
                        # connector itself the simulator lets the same two
                        # bodies run side by side.
                        if self.laterally_apart(a, b):
                            continue
                        # The vehicle on the downstream road (the connector)
                        # always leads. Comparing fronts alone let a long car
                        # still on the approach "lead" a motorbike already on
                        # the connector while the simulator's car following
                        # ordered them the other way: a permanent mutual wait.
                        a_down = not a.road.endswith("_in")
                        b_down = not b.road.endswith("_in")
                        if a_down != b_down:
                            lead, foll = (a, b) if a_down else (b, a)
                        else:
                            lead, foll = (a, b) if a.front >= b.front else (b, a)
                        if ((lead.units is not None or foll.units is not None) and
                                lead.front - lead.length / 2.0 <= foll.front):
                            # Beside a long rigid body, not ahead of it: more
                            # than half of the "leader" is alongside, so only
                            # the lateral checks apply. A stop at
                            # ``lead.back`` lay metres behind the truck's own
                            # nose (a stop from speed in one tick). A body
                            # mostly ahead stays a leader: across the seam
                            # jog the two paths converge.
                            continue
                        target = lead.back - foll.min_gap
                        if target < stop[foll.vid][0]:
                            stop[foll.vid] = (target, lead.speed)
                        self.stats["seam_follow"] += 1
                        decisions.append(PairDecision(lead.vid, foll.vid, None, "seam_follow"))
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
                        if mode == "ahead" and self.p.box_clear and not g.on_junction:
                            # With the box kept clear g is held at the stop
                            # line until the leftover has gone. Giving it the
                            # zone as well made each wait for the other.
                            mode = "wait"
                        if mode == "ahead":
                            # Lach doan: g slips through ahead of the leftover
                            # vehicle, which waits a moment if it must.
                            edges.append([g, c, [q.swapped() for q in seen_g], "weave"])
                            self.stats["weave"] += 1
                            continue
                        edges.append([c, g, seen_g, "clearing"])
                        self.clearing_holds[g.vid] = min(
                            c.speed, self.clearing_holds.get(g.vid, INF))
                        continue
                z = min(zones, key=lambda q: min(q.a_in - a.front, q.b_in - b.front))
                if a.arm == b.arm and z.kind == "diverge":
                    # Same approach, paths that separate: plain car following
                    # behind the leader's back until the paths have split.
                    # A wide predicted corridor can touch the next sublane
                    # even when the two bodies are currently side by side.
                    # Do not turn that future possibility into a same-lane
                    # follower stop tens of metres before the junction.
                    if (a.road.endswith("_in") and b.road.endswith("_in") and
                            abs(a.lateral_shift - b.lateral_shift) >=
                            a.half_width + b.half_width + 0.15):
                        continue
                    # One of them has crossed the line: lateral shifts are
                    # road-local, so measure in the plane. Bodies that are
                    # apart sideways on paths that only separate further are
                    # not leader and follower (a through vehicle at the line
                    # beside a left turner that has just entered).
                    if self.apart_in_plane(a, b):
                        continue
                    # As at the seam: a vehicle already on its connector is
                    # ahead of one still on the approach, whatever the path
                    # coordinates (rebased through the seam pose) say.
                    a_down = not a.road.endswith("_in")
                    b_down = not b.road.endswith("_in")
                    if a_down != b_down:
                        lead, foll = (a, b) if a_down else (b, a)
                    else:
                        # Order by body centre, like the simulator's own car
                        # following. Ordering by front made a short vehicle
                        # "follow" a long one that is beside or behind it.
                        lead, foll = ((a, b) if a.front - a.length / 2.0 >=
                                      b.front - b.length / 2.0 else (b, a))
                    target = lead.back - foll.min_gap
                    if target < stop[foll.vid][0]:
                        stop[foll.vid] = (target, lead.speed)
                    self.stats["diverge"] += 1
                    decisions.append(PairDecision(lead.vid, foll.vid, None, "diverge"))
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

        # Leftovers have absolute priority over a green vehicle that is not
        # free to go. A vehicle that must wait for one leftover may not at
        # the same time slip ahead of another: it would stand in the box
        # holding the second while the first is held by something else, and
        # the two phases lock (seed 1, fixed plan, tick ~1,500).
        for item in edges:
            if item[3] == "weave" and item[0].vid in self.clearing_holds:
                item[0], item[1] = item[1], item[0]
                item[2] = [q.swapped() for q in item[2]]
                item[3] = "clearing"
        self._break_cycles(edges)

        # Minor gate with creeping (nhich dan / cuop duong). A left turner does
        # not wait for one gap across the whole opposing stream: it enters the
        # junction and claims the conflict zones ONE BY ONE, in path order,
        # as each becomes takeable (gap, forcing a foe that can still brake,
        # or following its group). It stops just before the first zone it
        # cannot take, possibly in the middle of the stream, where through
        # vehicles of the sublanes it already blocks must stop or filter past.
        commits = set()
        minor_waits = []  # (blocking foe, waiting minor)
        # Who physically holds whom before the minor gate runs.
        self._hard_held = {}
        for first, second, _, reason in edges:
            if reason in ("in_zone", "both_in_zone", "clearing", "cannot_stop"):
                self._hard_held.setdefault(second.vid, []).append(first)
        active = [ag for ag in agents if not ag.passive and ag.vid not in pending_ids]
        first_over = {}   # foe vid -> minors that go ahead of it this tick
        claimed = []      # (minor, zones on its path it may enter this tick)
        for m in sorted(pending, key=lambda g: -g.front):
            items = []  # (zone on m's path, foe)
            m_corridor, m_arm, m_movement = m.corridor, m.arm, m.movement
            for f in active + [c for c, _ in claimed]:
                if f is m or f.corridor == m_corridor or f.arm == m_arm:
                    continue
                if not zone_lookup(m_movement, f.movement):
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
                if verdict != "ok" and mode == "clearing":
                    self.clearing_holds[m.vid] = min(
                        f.speed, self.clearing_holds.get(m.vid, INF))
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
                minor_waits.append((f, m))
            claimed.append((m, taken))
            if math.isfinite(target):
                self.stats["minor_wait"] += 1
                if target < stop[m.vid][0]:
                    stop[m.vid] = (target, 0.0)
                decisions.append(PairDecision(blocker.vid, m.vid, None, "minor_wait"))
            else:
                self.stats["minor_gap"] += 1

        self._release_blocked_minor_claims(edges, minor_waits, stop)

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

    def _release_blocked_minor_claims(self, edges, minor_waits, stop):
        """Drop a minor's claim on a zone when the claim closes a wait cycle.

        The minor gate runs after ``_break_cycles`` and lets a turning vehicle
        take conflict zones one by one ("ahead", "forced", "group"). A minor
        that is then held before a *later* zone keeps the foes of the zones
        it took waiting although it is not moving. When those foes are, in
        turn, what the minor waits for (through physical in-zone edges) the
        junction locks for good, e.g. after a long green: leftover through
        vehicles wait for a right turner that waits for them. In such a cycle
        the minor gives the zone back: the foe goes, the minor holds before
        the zone it has not entered yet.
        """
        claims = ("ahead", "forced", "group")
        for _ in range(len(edges) + 1):
            waits = {}
            for item in edges:
                waits.setdefault(item[1].vid, []).append((item[0].vid, item))
            for foe, minor in minor_waits:
                waits.setdefault(minor.vid, []).append((foe.vid, None))
            state, released = {}, False
            for root in list(waits):
                if state.get(root) is not None or released:
                    continue
                stack, path = [root], []
                iters = [iter(waits.get(root, ()))]
                state[root] = 1
                while stack and not released:
                    step = next(iters[-1], None)
                    if step is None:
                        state[stack.pop()] = 2
                        iters.pop()
                        if path:
                            path.pop()
                        continue
                    nxt, item = step
                    if state.get(nxt) == 1:
                        start = stack.index(nxt)
                        cycle_items = path[start:] + [item]
                        for candidate in cycle_items:
                            if candidate is not None and candidate[3] in claims:
                                minor, foe, zones = candidate[0], candidate[1], candidate[2]
                                edges.remove(candidate)
                                hold = min(z.b_in for z in zones) - minor.min_gap
                                if hold < stop[minor.vid][0]:
                                    stop[minor.vid] = (hold, 0.0)
                                minor_waits.append((foe, minor))
                                self.stats["claim_released"] = self.stats.get(
                                    "claim_released", 0) + 1
                                released = True
                                break
                    elif state.get(nxt) is None:
                        state[nxt] = 1
                        stack.append(nxt)
                        path.append(item)
                        iters.append(iter(waits.get(nxt, ())))
            if not released:
                return

    def _break_cycles(self, edges):
        """Flip non-physical edges until no wait-for cycle can be broken.

        One pass flips at most one edge per cycle it meets (the longest-held
        vehicle goes), but a flip can close another cycle through physical
        edges, so passes repeat until none flips. A flipped edge is final:
        it is never flipped back, which bounds the loop by the edge count.
        """
        fixed = ("in_zone", "both_in_zone", "diverge", "cannot_stop", "clearing",
                 "deadlock_break")
        for _ in range(len(edges) + 1):
            waits = {}
            for item in edges:
                waits.setdefault(item[1].vid, []).append(item)
            state = {}
            flipped = False
            # Iterative DFS: the wait-for graph of a saturated junction is
            # deeper than Python's recursion limit allows.
            for root in list(waits):
                if state.get(root) is not None or flipped:
                    continue
                stack = [root]
                iters = [iter(list(waits.get(root, ())))]
                state[root] = 1
                while stack and not flipped:
                    item = next(iters[-1], None)
                    if item is None:
                        state[stack.pop()] = 2
                        iters.pop()
                        continue
                    nxt = item[0].vid
                    if state.get(nxt) == 1:
                        # A list keeps the order independent of
                        # PYTHONHASHSEED (string vids in a set are not ordered).
                        cycle = stack[stack.index(nxt):]
                        members = set(cycle)
                        cands = [it for v in cycle for it in waits.get(v, ())
                                 if it[0].vid in members and it[3] not in fixed]
                        if cands:
                            e = max(cands, key=lambda it: (it[1].hold_time, it[1].front))
                            e[0], e[1] = e[1], e[0]
                            e[2] = [q.swapped() for q in e[2]]
                            e[3] = "deadlock_break"
                            self.stats["deadlock_break"] += 1
                            flipped = True
                        else:
                            stuck = [it for v in cycle for it in waits.get(v, ())
                                     if it[0].vid in members and
                                     it[3] in ("in_zone", "both_in_zone") and
                                     it[0].speed < 0.5 and it[1].speed < 0.5 and
                                     it[1].hold_time >= self.p.squeeze_after]
                            if stuck:
                                e = max(stuck, key=lambda it: (it[1].hold_time, it[1].front))
                                for index, item_ in enumerate(edges):
                                    if item_ is e:
                                        del edges[index]
                                        break
                                self.stats["squeeze"] = self.stats.get("squeeze", 0) + 1
                                flipped = True
                    elif state.get(nxt) is None:
                        state[nxt] = 1
                        stack.append(nxt)
                        iters.append(iter(list(waits.get(nxt, ()))))
            if not flipped:
                return


# ---------------------------------------------------------------------------
# Adapter for python_sim.LeVanLuongSimulator (safe mode)
# ---------------------------------------------------------------------------

class _SimulatorZoneMap(ZoneMap):
    def __getstate__(self):
        # SimulatorAdapter._sample uses immutable road geometry and the band
        # encoded in the movement ID. A resumed worker rebuilds these tables.
        return dict(vars(self), _cache={}, _oriented={}, _samples={})


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
        self.zone_map = _SimulatorZoneMap(self._sample, lateral_margin=lateral_margin)
        self.negotiator = JunctionNegotiator(self.zone_map, params)
        self.negotiator.seed = int(getattr(sim, "seed", 0))
        self.wait: dict[int, float] = {}
        self.hold: dict[int, float] = {}
        self.commit: dict[int, float] = {}
        self._seam_off: dict[int, float] = {}
        self._conflict_held: set[int] = set()
        self.hard_caps = 0
        self.creep_min = 0.2
        # A green vehicle within this distance (plus its length) of its wait
        # point inside the junction counts as waiting there.
        self.advance_min = 4.0
        # Upper bound of the corner sweep of a car body (m). The actual sweep
        # is computed per path from its curvature (``_sweep``): a straight
        # through path has none, a tight right turn up to this bound.
        self.sweep_car = 0.75
        self.sweep_moto = 0.0
        # Upper bound of the band widening of a two-axle heavy vehicle (m),
        # see ``_rigid_band``.
        self.sweep_heavy = 2.5
        self._rigid_band_cache: dict[tuple, tuple[float, float]] = {}
        self._sweep_cache: dict[tuple[str, int], float] = {}
        self.box_clear = self.negotiator.p.box_clear
        self.filter_enabled = True
        self.filter_cooldown = 1.0
        self.filter_moves = 0
        # Evasion bookkeeping: gap events of through leaders facing a left
        # turner (first time seen), and manoeuvres actually performed.
        self._evade_seen: set[tuple[str, str]] = set()
        self.evade_gap_events = {"moto": 0, "car": 0, "heavy": 0}
        self.evade_events: list[tuple] = []  # (time, vid, kind, motorbike, sublanes)

    def _sample(self, movement_id):
        if "#" in movement_id:
            # Continuous lateral band: "<connector>#<centre shift>:<half width>".
            base, spec = movement_id.split("#", 1)
            shift, half_width = (float(x) for x in spec.split(":"))
        else:
            base, spec = movement_id.split("@", 1)
            lane, occupied = (int(x) for x in spec.split(":"))
            shift = (lane + (occupied - 1) / 2.0) * 1.05
            half_width = self.sim._half_width_by_occupied[occupied]
        road = self.sim._geometry_roads[base]
        n = max(2, int(math.ceil(road.length / self.sample_step)) + 1)
        s = np.linspace(0.0, road.length, n)
        pts = np.array([(p.x, p.y) for p in (road.pose(float(x), shift) for x in s)])
        return PathSample(pts, s, half_width, road.length)

    def _body_half(self, v, base, shift):
        """Half width of the band a body sweeps on path ``base`` at ``shift``:
        half its width plus the swing of its corners on curves."""
        if v.occupied == 1:
            return v.half_width + self.sweep_moto
        if getattr(v, "wheelbase", 0.0) > 0.0:
            return v.half_width + self._rigid_band(v, base, shift)[1]
        return v.half_width + self._sweep(base, shift, v.length, self.sweep_car)

    def _rigid_band(self, v, base, shift):
        """(centre offset, half-width extra) of the band a two-axle rigid
        body sweeps when its front axle follows the path at ``shift``.

        On the tightest curve of the path, radius R of the front axle path,
        the rear axle runs at sqrt(R^2 - wb^2): the body reaches
        R - sqrt(R^2 - wb^2) inside the front axle path, and its outer front
        corner sqrt((R_rear + w/2)^2 + (wb + fo)^2) - (R + w/2) outside it.
        The band is shifted towards the inside of the turn by half the
        difference and widened by half the sum (+0.1 m). Shifts grow to the
        left, so a left (counter-clockwise) turn has its inside at +shift.
        """
        key = (base, int(round(shift / 0.35)), round(v.wheelbase, 2),
               round(v.front_overhang, 2), round(v.half_width, 3))
        cached = self._rigid_band_cache.get(key)
        if cached is not None:
            return cached
        road = self.sim._geometry_roads[base]
        step = 0.5
        n = max(3, int(road.length / step))
        poses = [road.pose(min(road.length, i * step), key[1] * 0.35) for i in range(n + 1)]
        window = 8  # 4 m of reference progress
        curvature, direction = 0.0, 1.0
        for i in range(len(poses) - window):
            a, b = poses[i], poses[i + window]
            cross = a.tx * b.ty - a.ty * b.tx
            turn = abs(math.atan2(cross, a.tx * b.tx + a.ty * b.ty))
            arc = sum(math.hypot(poses[j + 1].x - poses[j].x, poses[j + 1].y - poses[j].y)
                      for j in range(i, i + window))
            if arc > 1e-6 and turn / arc > curvature:
                curvature, direction = turn / arc, (1.0 if cross >= 0 else -1.0)
        wb, fo, hw = v.wheelbase, v.front_overhang, v.half_width
        if curvature < 1e-6:
            result = (0.0, 0.1)
        else:
            radius = max(1.0 / curvature, wb + 0.1)
            rear = math.sqrt(radius * radius - wb * wb)
            inner = radius - rear
            outer = max(0.0, math.hypot(rear + hw, wb + fo) - (radius + hw))
            result = (direction * (inner - outer) / 2.0,
                      min(self.sweep_heavy, (inner + outer) / 2.0 + 0.1))
        self._rigid_band_cache[key] = result
        return result

    def _sweep(self, base, shift, length, cap=None):
        """Corner sweep of a ``length`` m rectangle centred on a path.

        A rectangle kept tangent to a curve of radius R leaves the centre-line
        band by its sagitta (L/2)^2 / (2R) at the outer corners (SUMO adds the
        same term). R is the tightest radius of the path at this lateral
        shift, measured over a 2 m window; 0.1 m covers discretisation. A
        constant 0.75 m used to be added to every car path, straight ones
        included, so through cars held vehicles in the next sublane.
        """
        cap = self.sweep_car if cap is None else cap
        key = (base, int(round(shift / 0.35)), int(round(length * 10)), cap)
        cached = self._sweep_cache.get(key)
        if cached is not None:
            return cached
        road = self.sim._geometry_roads[base]
        step = 0.5
        n = max(3, int(road.length / step))
        poses = [road.pose(min(road.length, i * step), key[1] * 0.35) for i in range(n + 1)]
        window = 4  # 2 m of reference progress
        curvature = 0.0
        for i in range(len(poses) - window):
            a, b = poses[i], poses[i + window]
            turn = abs((math.atan2(b.ty, b.tx) - math.atan2(a.ty, a.tx) + math.pi)
                       % (2 * math.pi) - math.pi)
            # Arc length of THIS sublane (inner sublanes of a turn are shorter).
            arc = sum(math.hypot(poses[j + 1].x - poses[j].x, poses[j + 1].y - poses[j].y)
                      for j in range(i, i + window))
            if arc > 1e-6:
                curvature = max(curvature, turn / arc)
        sweep = min(cap, (length / 2.0) ** 2 * curvature / 2.0 + 0.1)
        self._sweep_cache[key] = sweep
        return sweep

    def _band_path(self, v, base, on_conn):
        """Lateral band the vehicle will actually sweep on its connector.

        Lane indices are not enough: crossing the stop-line seam can rebase the
        lateral shift and the vehicle then drifts to its target sublane, so the
        sublane path of its lane index may be metres away from its body.
        """
        sim = self.sim
        # Body half width plus the sweep of a rectangle on a curved path
        # (front/rear corners leave the centre-line band; SUMO adds a sagitta).
        rigid_offset = 0.0
        if v.occupied == 1:
            half = self._body_half(v, base, 0.0)
        else:
            ref = (v.current_lateral_shift if on_conn else
                   sim._continuous_seam_pose(f"{v.arm}_in", base, v.current_lateral_shift)[1])
            if getattr(v, "wheelbase", 0.0) > 0.0 and not on_conn:
                # It will drive the sublane it is mapped to (a heavy vehicle
                # swings out on a right turn).
                lane = sim._mapped_lane(v, v.lane, f"{v.arm}_in", base)
                ref = (lane + (v.occupied - 1) / 2.0) * 1.05
            half = self._body_half(v, base, ref)
            if getattr(v, "wheelbase", 0.0) > 0.0:
                rigid_offset = self._rigid_band(v, base, ref)[0]
        if on_conn and getattr(v, "evading", False) and v.lane_change_progress < 1.0:
            # Evasive move checked when it was started: the vehicle reaches
            # every remaining conflict already at its target shift.
            lo = hi = v.target_lateral_shift
        elif on_conn:
            if getattr(v, "evading", False):
                v.evading = False
            lo = min(v.current_lateral_shift, v.target_lateral_shift)
            hi = max(v.current_lateral_shift, v.target_lateral_shift)
        else:
            shift = sim._continuous_seam_pose(f"{v.arm}_in", base, v.current_lateral_shift)[1]
            lane = sim._mapped_lane(v, v.lane, f"{v.arm}_in", base)
            target = (lane + (v.occupied - 1) / 2.0) * 1.05
            n_lanes = sim.roads[base]["num_lanes"]
            min_shift = (v.occupied - 1) / 2.0 * 1.05
            max_shift = (n_lanes - 1 - (v.occupied - 1) / 2.0) * 1.05
            if (min_shift - 0.05 <= shift <= max_shift + 0.05 and
                    not (getattr(v, "wheelbase", 0.0) > 0.0 and abs(shift - target) > 0.3)):
                lo = hi = shift
            else:
                lo, hi = min(shift, target), max(shift, target)
        lo, hi = lo + rigid_offset, hi + rigid_offset
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
            # Time held by a conflict. It keeps running while the vehicle
            # stays (almost) still: a wait that drops out of the decisions
            # for a tick (a squeeze release that a body in the way made
            # useless) must not restart the clock, or the patience rules
            # that depend on it (reclaim, squeeze) never fire again.
            previous = self.hold.get(vid, 0.0)
            held = v.speed < 1.0 and (vid in self._conflict_held or previous > 0.0)
            self.hold[vid] = previous + dt if held else 0.0
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
            seam_pose = (None if on_conn else sim._continuous_seam_pose(
                f"{v.arm}_in", base, v.current_lateral_shift))
            seam_off = 0.0 if on_conn else seam_pose[0]
            seam_shift = v.current_lateral_shift if on_conn else seam_pose[1]
            pose = sim._geometry_roads[road].pose(
                sim._road_state(v)[1], v.current_lateral_shift)
            self._seam_off[vid] = seam_off
            front = v.position + v.length / 2.0 - v.approach_end + seam_off
            ag = Agent(vid=str(vid), movement=movement, arm=v.arm, turn=v.maneuver,
                       front=front, length=v.length, speed=max(0.0, v.speed),
                       accel=v.accel, decel=v.decel,
                       max_decel=min(7.5, 2.0 * v.decel), min_gap=max(0.5, v.gap),
                       desired=min(v.desired, 40.0 / 3.6), motorbike=v.occupied == 1,
                       on_junction=on_conn or v.position + v.length / 2.0 > v.approach_end,
                       arm_green=green,
                       lateral_shift=v.current_lateral_shift,
                       half_width=sim._half_width(v),
                       connector_shift=(v.current_lateral_shift if on_conn else
                                        seam_shift),
                       connector_span=((min(v.current_lateral_shift, v.target_lateral_shift),
                                        max(v.current_lateral_shift, v.target_lateral_shift))
                                       if on_conn else (seam_shift, seam_shift)),
                       xy=(pose.x, pose.y), heading=(pose.tx, pose.ty),
                       units=(sim._body_axis_points(v)
                              if getattr(v, "wheelbase", 0.0) > 0.0 else None),
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
            slow_prefix = [0]
            for ag in members:
                slow_prefix.append(slow_prefix[-1] + (ag.speed < 4.0))
            end = 1
            for i, ag in enumerate(members):
                end = max(end, i + 1)
                while end < len(members) and ag.front - members[end].front <= self.group_window:
                    end += 1
                ag.group_size = 1 + slow_prefix[end] - slow_prefix[i + 1]
            # Vu & Shimizu (2010) group: a vehicle joins when it is within
            # l0 of the nearest member (bumper gap) and moves the same way
            # (same connector). MC1 counts the motorcycles of the group.
            chain, chains = [], []
            min_back = math.inf
            for ag in members:
                if chain and min_back - ag.front > self.negotiator.p.group_gap:
                    chains.append(chain)
                    chain = []
                    min_back = math.inf
                chain.append(ag)
                min_back = min(min_back, ag.back)
            if chain:
                chains.append(chain)
            for chain in chains:
                motos = sum(1 for other in chain if other.motorbike)
                for ag in chain:
                    ag.group_moto = max(1, motos)
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
        seam_followers = {d.second for d in self.negotiator.last_decisions
                          if d.reason in ("seam_follow", "diverge")}
        seam_targets = {}
        for decision in self.negotiator.last_decisions:
            if decision.reason != "seam_follow":
                continue
            leader = by_vid.get(decision.first)
            follower = by_vid.get(decision.second)
            if (leader is None or follower is None or
                    leader.arm != follower.arm or leader.to_arm != follower.to_arm or
                    leader.position <= follower.position):
                continue
            if ((leader.wheelbase > 0.0 or follower.wheelbase > 0.0) and
                    leader.position <= follower.position + follower.length / 2.0):
                # Beside a long body, not ahead of it: with a 9 m truck the
                # centres say little about order. When the other centre is
                # still alongside the truck, the truck's band envelope
                # touches the other body's band, the bodies do not (lateral
                # checks keep them apart). Taking its rear as the stop put
                # the stop metres behind the truck's own nose.
                continue
            stopping_front = (leader.position - leader.length / 2.0 -
                              follower.gap)
            target = (stopping_front - follower.approach_end +
                      self._seam_off.get(follower.creation_id, 0.0))
            # Several vehicles can lead across the seam: the nearest governs.
            # (Each used to overwrite the previous one, so the last in the
            # list decided, whichever it was.)
            if decision.second not in seam_targets or target < seam_targets[decision.second][0]:
                seam_targets[decision.second] = (target, leader.speed)
        for vid, (target, speed) in seam_targets.items():
            # Replace the negotiator's static seam stop by the moving bumper,
            # unless a crossing conflict holds the vehicle clearly earlier
            # (the seam stop itself lies within min_gap - gap of ``target``).
            if stop[vid][0] >= target - 0.6:
                stop[vid] = (target, speed)
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
        # Only vehicles that will actually cross count: one on its connector,
        # or one past the line that holds an entry token. A red-arm vehicle
        # standing with its nose over the line is not clearing anything and
        # must not keep every conflicting approach closed.
        # The hold follows the negotiator's own verdict: a vehicle waits at
        # the line only when it would otherwise have to stop inside the box
        # for that leftover. A leftover that leaves the shared zone before
        # the vehicle can reach it holds nobody (it used to: any shared zone
        # not yet cleared by both was enough, however far apart in time).
        blocked_by_clearing = set()
        if self.box_clear:
            holds = self.negotiator.clearing_holds
            # Green vehicles already waiting inside the box for a leftover.
            # A vehicle that is still rolling towards a distant wait point is
            # not waiting yet and closes nothing behind it.
            waiting_inside = [ag for ag in agents
                              if ag.on_junction and ag.arm_green and ag.vid in holds
                              and (ag.speed < 1.0 or
                                   stop[ag.vid][0] - ag.front < self.advance_min + ag.length)]
            for ag in agents:
                if ag.on_junction or ag.passive or not ag.arm_green:
                    continue
                # Nobody queues behind a vehicle that waits in the box: the
                # queue would stand across the paths of other leftovers.
                # Only the lateral band of that vehicle is closed; the other
                # sublanes of the stream stay open.
                if any(w.corridor == ag.corridor and
                       not self.negotiator.laterally_apart(ag, w)
                       for w in waiting_inside):
                    blocked_by_clearing.add(ag.vid)
                    continue
                if ag.vid in holds:
                    # It has to yield to a leftover: it does so at the line.
                    # Moving up to a wait point inside the junction was tried
                    # and locked the box (random policy, seed 1): the waiting
                    # bodies end up in the paths of other leftovers.
                    blocked_by_clearing.add(ag.vid)
        storage = {}
        for ag in agents:
            if ag.on_junction and ag.turn == "left" and ag.commit_time is None:
                # Motorbike equivalents: a car counts 2, a 9 m heavy vehicle 4.
                storage[ag.arm] = storage.get(ag.arm, 0) + (
                    1 if ag.motorbike else max(2, int(round(ag.length / 2.25))))
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
            # A stop target that is only the moving rear bumper of the
            # vehicle ahead on the same path is ordinary car following. It
            # must not deny entry: that halted every follower at the line for
            # a tick or two while its leader pulled away (stop-and-go
            # discharge and one-tick stops from crawl speed).
            following = obstacle_speed > 0.5 and ag.vid in seam_followers
            # A truck or bus granted entry a moment ago cannot always stop
            # on the line when the grant is withdrawn (a foe enters the zone
            # it will cross later): its hard braking is about 4.2 m/s^2
            # (passengers standing, load), so stopping metres before the
            # line from 3.6 m/s is not possible. It keeps its entry and
            # brakes to the negotiator's stop point before the conflict
            # zone instead, which it can reach (SUMO MSLink: an approaching
            # vehicle that cannot brake in time keeps its link).
            brake_through = False
            if (v.wheelbase > 0.0 and v.creation_id in sim._junction_grants and
                    v.speed > 0.5 and math.isfinite(s) and not ag.passive):
                hard = max(4.2, 1.4 * v.decel)
                to_line = sim._stop_position(v) - v.position
                front = v.position + v.length / 2.0 - v.approach_end
                brake_through = (
                    v.speed * v.speed > 2.0 * hard * max(0.0, to_line) and
                    s - front >= v.speed * v.speed / (2.0 * hard))
            if brake_through or ((s > self.creep_min or not math.isfinite(s) or following)
                    and room_ok and store_ok
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
        if self.negotiator.p.evade:
            self._evade(agents, by_vid, stop, now)
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
                    if max(lo - ohi, olo - hi, 0.0) < v.half_width + o.half_width + 0.1:
                        room = False
                        break
                if not room:
                    continue
                q = 0.15
                centre = round(target / q) * q
                cand = f"{road}#{centre:.2f}:{v.half_width + self.sweep_moto + math.ceil(abs(target - centre) / 0.1) * 0.1:.2f}"
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

    # ------------------------------------------------------------ evasion
    def _target_band(self, v, road, target):
        half = self._body_half(v, road, target)
        q = 0.15
        centre = round(target / q) * q
        extra = math.ceil(abs(target - centre) / 0.1) * 0.1
        return f"{road}#{centre:.2f}:{half + extra:.2f}"

    def _evade_clear(self, alt, ego, main, agents, d_move, swept, force=False):
        """Can ``alt`` (ego in its target band) go on without stopping?

        Returns (ok, kind) with kind "cut_tail" / "run_in_front" relative to
        the left turner ``main`` that holds ego now, or "swerve" when the new
        band no longer meets its path at all. With ``force`` the left turner
        ``main`` may be made to give way (kind "run_in_front_forced") if it
        has not committed and can still brake comfortably. When not ok, kind
        describes the blocking conflict (for diagnostics).
        """
        neg = self.negotiator
        # The same margins the negotiation itself grants: a time gap to pass
        # ahead of a vehicle, the follow gap to pass behind it.
        ahead_gap = max(self.negotiator.p.evade_pet, neg.timegap(ego))
        behind_gap = neg.follow_gap(ego)
        kind = None
        for g in agents:
            if g is ego or g.corridor == ego.corridor or (g.passive and not g.on_junction):
                continue
            checks = [(alt, z) for z in self.zone_map.zones(alt.movement, g.movement)
                      if alt.back < z.a_out and g.back < z.b_out]
            # conflicts reached while the body is still moving sideways
            checks += [(swept, z) for z in self.zone_map.zones(swept.movement, g.movement)
                       if swept.back < z.a_out and g.back < z.b_out and
                       z.a_in < ego.front + d_move]
            # A crossing minor that has not committed, has not accepted a gap
            # in front of ego and can still brake will yield to ego as it
            # yields to any major (only the left turner that holds ego now
            # is passed by timing alone).
            yields = (g is not main and neg.is_minor(g) and g.commit_time is None and
                      (g.vid, ego.vid) not in neg._accepted)
            forceable = (force and g is main and g.commit_time is None and
                         not g.passive)
            for who, z in checks:
                if neg.in_zone(g, z.b_in, z.b_out) and g.speed < 0.5:
                    return False, ("stalled", g is main, g.turn, who is swept)
                if (yields and not neg.in_zone(g, z.b_in, z.b_out) and
                        neg.can_stop(g, z.b_in, 0.6 * g.max_decel)):
                    continue
                e_in, e_out = neg.arrival(who, z.a_in), neg.leave(who, z.a_out)
                g_in, g_out = neg.arrival(g, z.b_in), neg.leave(g, z.b_out)
                if neg.in_zone(g, z.b_in, z.b_out):
                    g_in = 0.0
                if e_out + ahead_gap <= g_in:
                    side = "run_in_front"
                elif e_in >= g_out + behind_gap:
                    side = "cut_tail"
                elif (forceable and not neg.in_zone(g, z.b_in, z.b_out) and
                      neg.can_stop(g, z.b_in, self.negotiator.p.evade_force_decel_share *
                                   g.max_decel)):
                    side = "run_in_front_forced"
                else:
                    return False, ("timing", g is main, g.turn, who is swept,
                                   neg.in_zone(g, z.b_in, z.b_out), g.speed < 0.5)
                if g is main and who is alt:
                    if side == "run_in_front_forced" or kind == "run_in_front_forced":
                        kind = "run_in_front_forced"
                    elif kind is None or kind == side:
                        kind = side
                    else:
                        kind = "cut_tail"
        if kind is None:
            passed = any(z.b_out <= main.back for z in
                         self.zone_map.zones(alt.movement, main.movement))
            kind = "cut_tail" if passed else "swerve"
        return True, kind

    def _evade(self, agents, by_vid, stop, now):
        """Cut tail / run in front for straight-go group leaders.

        A through vehicle inside the junction that has to brake for a
        crossing left turner may, with the propensity of Vu & Shimizu (2010),
        move sideways to a band where it passes behind the left turner's tail
        or ahead of it without stopping. Candidate bands are 1..3 sublanes
        either side; the one needing the smallest deviation wins (maximum
        progress along the desired direction, Trinh et al. 2018). The lateral
        speed stays inside a heading cone (motorcycles +-30 deg as Trinh et
        al. 2018, cars +-0.2 rad as Kanagaraj & Treiber 2018), so a slow
        vehicle can only shift a little before the conflict.
        """
        sim = self.sim
        p = self.negotiator.p
        ag_by_vid = {ag.vid: ag for ag in agents}
        holders = {}
        for d in self.negotiator.last_decisions:
            if d.reason not in ("seam_follow", "diverge"):
                holders.setdefault(d.second, []).append(d.first)
        live = set(ag_by_vid)
        self._evade_seen = {k for k in self._evade_seen if k[0] in live}
        for ag in agents:
            if ag.turn != "through" or not ag.on_junction or ag.vid not in holders:
                continue
            v = by_vid[ag.vid]
            road = v.conn_road
            if (v.road != road or v.lane_change_progress < 1.0 or
                    v.speed < p.evade_min_speed or v.leading_distance < p.evade_leader_gap):
                continue
            s_stop = stop[ag.vid][0]
            if not math.isfinite(s_stop):
                continue
            if s_stop - ag.front > v.speed * v.speed / (2.0 * max(v.decel, 0.1)) + 2.0 * v.speed + 3.0:
                continue  # not affected yet
            mains = [ag_by_vid[f] for f in holders[ag.vid]
                     if f in ag_by_vid and ag_by_vid[f].turn == "left"]
            if not mains:
                continue
            main = min(mains, key=lambda g: abs(g.front))
            key = (ag.vid, main.vid)
            if key not in self._evade_seen:
                self._evade_seen.add(key)
                self.evade_gap_events["moto" if ag.motorbike else v.kind] += 1
            if ag.motorbike:
                p_cut, p_front = p.evade_cut_moto, p.evade_front_moto
            elif v.kind == "car":
                p_cut, p_front = p.evade_cut_car, p.evade_front_car
            else:
                p_cut, p_front = p.evade_cut_heavy, p.evade_front_heavy
            draw = self.negotiator._draw
            will_cut = draw("cut:" + ag.vid, main.vid) < p_cut
            will_front = draw("front:" + ag.vid, main.vid) < p_front
            if not (will_cut or will_front):
                continue
            allowed = set()
            if will_cut:
                allowed |= {"cut_tail", "swerve"}
            if will_front:
                allowed |= {"run_in_front", "run_in_front_forced"}
            n_lanes = sim.roads[road]["num_lanes"]
            cur = v.current_lateral_shift
            lat_cap = 1.75 if v.occupied == 1 else 0.9
            cone = p.lateral_cone_moto if v.occupied == 1 else p.lateral_cone_car
            lat_speed = min(lat_cap, v.speed * math.tan(cone))
            prog = v.position - v.approach_end
            best = None
            for k in range(1, p.evade_max_sublanes + 1):
                for dl in (k, -k):
                    lane = v.lane + dl
                    if not 0 <= lane <= n_lanes - v.occupied:
                        continue
                    target = (lane + (v.occupied - 1) / 2.0) * 1.05
                    dist = abs(target - cur)
                    # Bezier lateral profile: peak rate 1.5 x mean rate.
                    duration = max(0.4, 1.5 * dist / max(lat_speed, 1e-6))
                    d_move = v.speed * duration
                    lo, hi = min(cur, target), max(cur, target)
                    room = True
                    for o in sim.vehicles:
                        if o is v or o.road != road:
                            continue
                        op = o.position - o.approach_end
                        ahead = op - prog
                        behind_reach = 1.0 + max(0.0, o.speed - v.speed) * duration
                        if not (-behind_reach - (o.length + v.length) / 2 < ahead <
                                d_move + 2.0 + (o.length + v.length) / 2):
                            continue
                        olo = min(o.current_lateral_shift, o.target_lateral_shift)
                        ohi = max(o.current_lateral_shift, o.target_lateral_shift)
                        if max(lo - ohi, olo - hi, 0.0) < (sim._half_width(v) +
                                                           sim._half_width(o) + 0.1):
                            room = False
                            break
                    if not room:
                        continue
                    alt = replace(ag, movement=self._target_band(v, road, target),
                                  connector_shift=target, connector_span=(target, target))
                    q = 0.15
                    centre = round((lo + hi) / 2.0 / q) * q
                    extra = math.ceil(((hi - lo) / 2.0 + abs((lo + hi) / 2.0 - centre)) / 0.1) * 0.1
                    half = (self._body_half(v, road, cur) if v.occupied > 1
                            else v.half_width)
                    swept = replace(ag, movement=f"{road}#{centre:.2f}:{half + extra:.2f}",
                                    connector_shift=(lo + hi) / 2.0, connector_span=(lo, hi))
                    ok, kind = self._evade_clear(alt, ag, main, agents, d_move, swept,
                                                 force=will_front)
                    if ok and kind in allowed:
                        best = (lane, target, duration, kind, k)
                        break
                if best is not None:
                    break
            if best is None:
                continue
            lane, target, duration, kind, k = best
            v.lane_change_from = v.lane
            v.lane = lane
            v.start_lateral_shift = cur
            v.target_lateral_shift = target
            v.lane_change_progress = 0.0
            v.time_since_lane_change = 0.0
            v.lateral_duration = duration
            v.evading = True
            if kind == "run_in_front_forced":
                self.negotiator._conceded.add((main.vid, ag.vid))
                kind = "run_in_front"
            self.negotiator.stats[kind] = self.negotiator.stats.get(kind, 0) + 1
            self.evade_events.append((now, v.creation_id, kind, v.occupied == 1, k))


if __package__:
    from ._native import load_if_built
    load_if_built(__name__, __file__)
