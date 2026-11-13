"""Pure reference for GAMA 2025.6.4's IDM and MOBIL decisions.

Source of truth: GAMA tag ``2025.06.4`` (commit
``7cde6fac892138a161d75c7d02d112a981cbfcb6``). In
``gama.extension.traffic/.../carfollowing/IDM.java`` lines 45-58 the
engine uses the standard IDM equation with agent-provided T, a, b, v0,
s0 and delta. In ``MOBIL.java`` lines 84-92 it scans the bounded lane
range in ascending order; lines 141-200 calculate candidate/follower
accelerations, reject follower acceleration <= -bSave, add right/left
bias, and retain only a strictly better incentive. Lines 114-122 and
207-212 govern cooldown updates.

Engine defaults are declared by tagged ``DrivingSkill.java`` lines 71-104:
s0=.5, speed coefficient=1, max speed=50, T=1.5, a=.3, b=3, delta=4,
politeness=.5, bSave=4, aThr=.2, aBias=.25, cooldown=4 and lane-change
limit=1. The Le Van Luong model overrides relevant values in
``models/LeVanLuong.gaml`` lines 230-245 (including T, s0, a, aThr and p).
Tagged ``Utils.java`` lines 96-173 returns a 1e6 m distance when no leader
exists. This port accepts that exact sentinel from callers; using ``math.inf``
is a free-road abstraction that omits the tiny interaction term
``(sStar / 1e6) ** 2``.

This module models the engine calculation only. It intentionally does not
implement Python simulator safety filters (front/back required-gap checks,
near-stop-line/speed gates, or mandatory-turn incentive bonuses).
Distances are bumper gaps in metres, speeds m/s, acceleration m/s^2, and
time seconds, as in the GAML model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Hashable, Mapping, Optional


@dataclass(frozen=True)
class IDMParams:
    """DrivingSkill defaults (model overrides should be supplied explicitly)."""
    time_headway: float = 1.5
    max_acceleration: float = 0.3
    max_deceleration: float = 3.0
    desired_speed: float = 50.0
    min_safety_distance: float = 0.5
    delta: float = 4.0
    speed_coeff: float = 1.0


def idm_acceleration(*, speed: float, leading_speed: float, gap: float,
                     params: IDMParams, road_max_speed: float = math.inf) -> float:
    """Port of tag IDM.java:45-58; GAMA's no-leader gap is 1e6 m (Utils.java:173).

    ``gap=inf`` is also accepted as a free-road abstraction.
    """
    a, b = params.max_acceleration, params.max_deceleration
    v0 = min(params.desired_speed, params.speed_coeff * road_max_speed)
    dv = speed - leading_speed
    s_star = params.min_safety_distance + max(
        0.0, speed * params.time_headway + speed * dv / (2.0 * math.sqrt(a * b)))
    free = (speed / v0) ** params.delta
    if gap == 0.0:
        interaction = math.inf if s_star != 0 else math.nan
    else:
        interaction = (s_star / gap) ** 2
    return a * (1.0 - free - interaction)


@dataclass(frozen=True)
class Candidate:
    """Neighbor data for a single lane, in GAMA findLeader/findFollower terms."""
    front_gap: float
    leader_speed: float = 0.0
    leader_same_direction: bool = True
    follower_id: Optional[Hashable] = None
    back_gap: float = math.inf
    follower_speed: float = 0.0
    follower_leading_distance: float = math.inf
    follower_params: IDMParams = field(default_factory=IDMParams)
    follower_road_max_speed: float = math.inf


@dataclass(frozen=True)
class MobilParams:
    politeness: float = 0.5
    safe_deceleration: float = 4.0
    acceleration_threshold: float = 0.2
    acceleration_bias: float = 0.25
    right_side_driving: bool = True
    lane_change_limit: int = 1
    cooldown: float = 4.0


@dataclass(frozen=True)
class MobilResult:
    lane: int
    acceleration: float
    time_since_lane_change: float
    incentive: float = 0.0


def choose_lane(*, current_lane: int, lane_count: int, lanes_occupied: int,
                speed: float, vehicle_length: float, current_gap: float,
                current_leader_speed: float, current_follower_id: Optional[Hashable],
                current_time_since_lane_change: float, timestep: float,
                vehicle_params: IDMParams, candidates: Mapping[int, Candidate],
                mobil: MobilParams = MobilParams(), road_max_speed: float = math.inf,
                allowed_lanes: tuple[int, ...] = (), leader_is_intersection: bool = False,
                linked_lane_limit: int = 0) -> MobilResult:
    """Port of GAMA MOBIL.chooseLane; candidate map supplies lane neighbors.

    ``candidates`` need only include lanes in the road's range. Missing entries
    are skipped. GAMA's optional linked-road random gate is not modeled; callers
    should provide already eligible lanes (the Le Van Luong GAML sets its linked
    lane limit/probability to zero).
    """
    upper_road = lane_count + linked_lane_limit - lanes_occupied
    lower = max(0, current_lane - mobil.lane_change_limit) if mobil.lane_change_limit != -1 else 0
    upper = min(upper_road, current_lane + mobil.lane_change_limit) if mobil.lane_change_limit != -1 else upper_road
    if current_gap < 0:
        stay_acc = -float.fromhex('0x1.fffffffffffffp+1023')
    else:
        stay_acc = idm_acceleration(speed=speed, leading_speed=current_leader_speed,
                                    gap=current_gap, params=vehicle_params,
                                    road_max_speed=road_max_speed)
        # Tag MOBIL.java:114-122 places this gate inside the non-overlap branch.
        # A negative current gap bypasses both cooldown and intersection checks.
        if leader_is_intersection or current_time_since_lane_change < mobil.cooldown:
            return MobilResult(current_lane, stay_acc,
                               current_time_since_lane_change + timestep)

    best_lane, best_acc, best_gain = current_lane, stay_acc, 0.0
    for lane in range(lower, upper + 1):
        if lane == current_lane or (allowed_lanes and lane not in allowed_lanes):
            continue
        c = candidates.get(lane)
        if c is None or c.front_gap < 0 or c.back_gap < 0:
            continue
        leader_speed = c.leader_speed if c.leader_same_direction else -c.leader_speed
        change_acc = idm_acceleration(speed=speed, leading_speed=leader_speed,
                                      gap=c.front_gap, params=vehicle_params,
                                      road_max_speed=road_max_speed)
        stay_follower_acc = change_follower_acc = 0.0
        if (c.follower_id is not None and c.follower_id != current_follower_id
                and c.follower_leading_distance >= c.back_gap):
            prior_gap = c.back_gap + vehicle_length + c.front_gap
            stay_follower_acc = idm_acceleration(
                speed=c.follower_speed, leading_speed=leader_speed, gap=prior_gap,
                params=c.follower_params, road_max_speed=c.follower_road_max_speed)
            change_follower_acc = idm_acceleration(
                speed=c.follower_speed, leading_speed=speed, gap=c.back_gap,
                params=c.follower_params, road_max_speed=c.follower_road_max_speed)
        if change_follower_acc <= -mobil.safe_deceleration:
            continue
        bias_cond = lane < current_lane if mobil.right_side_driving else lane > current_lane
        bias_sign = 1 if bias_cond else -1
        incentive = (change_acc - stay_acc
                     + mobil.politeness * (change_follower_acc - stay_follower_acc)
                     + mobil.acceleration_bias * bias_sign)
        # Java strict comparisons preserve the first (lowest-index) exact tie.
        if incentive > mobil.acceleration_threshold and incentive > best_gain:
            best_gain, best_lane, best_acc = incentive, lane, change_acc
    elapsed = 0.0 if best_lane != current_lane else current_time_since_lane_change + timestep
    return MobilResult(best_lane, best_acc, elapsed, best_gain)
