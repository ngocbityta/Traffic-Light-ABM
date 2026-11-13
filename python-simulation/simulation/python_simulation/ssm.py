"""Offline surrogate safety measures (SSM) from recorded trajectories.

Input: the column dict produced by :class:`trajectory.TrajectoryRecorder`
(``arrays()`` or ``trajectory.load(path)``). Everything is vectorised with
numpy; no simulator state is needed except for the conflict-point geometry.

Measures (see Traffic_Papers/TTC_*_Cum_*.md for the background):

* Rear-end, per follower/leader pair and tick (same road, laterally
  overlapping bodies; leader = smallest bumper gap ahead):
  - ``ttc``  classical 1-D TTC = D / dv (Hayward 1971)
  - ``mttc`` acceleration-aware TTC = smallest t > 0 of
    0.5*da*t^2 + dv*t - D = 0 (Ozbay et al. 2008; "TTC2" in Cum 1)
  - ``time_gap`` D / v_follower (Vogel 2003)
  - ``drac`` dv^2 / (2 D) (Cooper & Ferguson 1976)
  - ``cs`` lateral centre offset (Das & Maurya 2020)
* Crossing conflicts on connectors: ``pet`` post-encroachment time at each
  lane-level conflict point (Allen et al. 1978), entry/exit interpolated
  between ticks.
* Junction 2-D TTC with area-equivalent discs and constant velocity vectors
  (Mowafi & Saccomanno 2014 style). Straight-line extrapolation over-reports
  turning conflicts; use it as a screening measure next to PET.
* Exposure: TET and TIT for a TTC threshold (Minderhoud & Bovy 2001).

Thresholds are parameters. Defaults are the common literature values
(TTC* 1.5 s, PET 1.0/1.5 s, DRAC 3.4 m/s^2 comfortable). Type- and
offset-dependent thresholds (Das & Maurya 2020) were estimated on Indian
urban car-follower data with no motorbike followers; pass your own
``ttc_threshold`` function if you want to use them.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_TTC_THRESHOLD = 1.5
PET_CRITICAL = 1.0
PET_SERIOUS = 1.5
DRAC_COMFORT = 3.4
DRAC_HARD = 5.0
HARD_BRAKE = -4.0


# ---------------------------------------------------------------- formulas
def ttc_1d(gap, dv):
    """Classical TTC; inf where the follower is not faster."""
    gap = np.asarray(gap, dtype=float)
    dv = np.asarray(dv, dtype=float)
    out = np.full(np.broadcast(gap, dv).shape, np.inf)
    mask = dv > 1e-9
    np.divide(gap, dv, out=out, where=mask)
    return out


def mttc(gap, dv, da):
    """Smallest positive root of 0.5*da*t^2 + dv*t - gap = 0 (inf if none)."""
    gap, dv, da = np.broadcast_arrays(np.asarray(gap, float), np.asarray(dv, float),
                                      np.asarray(da, float))
    out = np.full(gap.shape, np.inf)
    linear = np.abs(da) < 1e-9
    lin_ok = linear & (dv > 1e-9)
    out[lin_ok] = gap[lin_ok] / dv[lin_ok]
    quad = ~linear
    a = 0.5 * da[quad]
    disc = dv[quad] ** 2 + 2.0 * da[quad] * gap[quad]
    roots = np.full(a.shape, np.inf)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    r1 = (-dv[quad] - sq) / (2 * a)
    r2 = (-dv[quad] + sq) / (2 * a)
    r1 = np.where(ok & (r1 > 1e-9), r1, np.inf)
    r2 = np.where(ok & (r2 > 1e-9), r2, np.inf)
    roots = np.minimum(r1, r2)
    out[quad] = roots
    out[gap <= 0] = 0.0
    return out


def drac(gap, dv):
    gap = np.asarray(gap, float)
    dv = np.asarray(dv, float)
    out = np.zeros(np.broadcast(gap, dv).shape)
    mask = (dv > 1e-9) & (gap > 1e-9)
    out[mask] = dv[mask] ** 2 / (2.0 * gap[mask])
    out[(dv > 1e-9) & (gap <= 1e-9)] = np.inf
    return out


def ttc_circle(p1, v1, r1, p2, v2, r2):
    """Constant-velocity time until two discs touch (0 if overlapping)."""
    dp = np.asarray(p1, float) - np.asarray(p2, float)
    dv = np.asarray(v1, float) - np.asarray(v2, float)
    r = np.asarray(r1, float) + np.asarray(r2, float)
    a = np.sum(dv * dv, axis=-1)
    b = 2.0 * np.sum(dp * dv, axis=-1)
    c = np.sum(dp * dp, axis=-1) - r * r
    out = np.full(np.shape(c), np.inf)
    overlap = c <= 0
    disc = b * b - 4 * a * c
    ok = (~overlap) & (a > 1e-12) & (disc >= 0)
    t = np.where(ok, (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2 * np.where(ok, a, 1.0)), np.inf)
    out = np.where(ok & (t > 0), t, out)
    out = np.where(overlap, 0.0, out)
    return out


# --------------------------------------------------------------- rear-end
def rear_end_pairs(data, max_ahead=12):
    """Follower/leader pairs per tick: dict of aligned arrays."""
    cycle, road, prog = data["cycle"], data["road"], data["progress"]
    order = np.lexsort((prog, road, cycle))
    c, r, p = cycle[order], road[order], prog[order]
    shift = data["shift"][order]
    hw = data["half_width"][order]
    length = data["length"][order]
    speed = data["speed"][order]
    accel = data["accel"][order]
    n = len(order)
    best_gap = np.full(n, np.inf)
    best_j = np.full(n, -1)
    for k in range(1, max_ahead + 1):
        i = np.arange(n - k)
        j = i + k
        same = (c[i] == c[j]) & (r[i] == r[j])
        lateral = np.abs(shift[i] - shift[j]) < (hw[i] + hw[j])
        gap = p[j] - p[i] - (length[i] + length[j]) / 2.0
        better = same & lateral & (gap < best_gap[i])
        best_gap[i[better]] = gap[better]
        best_j[i[better]] = j[better]
    f = np.nonzero(best_j >= 0)[0]
    l = best_j[f]
    gap = best_gap[f]
    dv = speed[f] - speed[l]
    da = accel[f] - accel[l]
    with np.errstate(divide="ignore", invalid="ignore"):
        time_gap = np.where(speed[f] > 0.1, gap / np.maximum(speed[f], 1e-9), np.inf)
    return {
        "row_follower": order[f], "row_leader": order[l], "cycle": c[f],
        "vid_follower": data["vid"][order[f]], "vid_leader": data["vid"][order[l]],
        "follower_is_car": data["is_car"][order[f]], "leader_is_car": data["is_car"][order[l]],
        "gap": gap, "dv": dv, "da": da, "cs": np.abs(shift[f] - shift[l]),
        "ttc": ttc_1d(gap, dv), "mttc": mttc(gap, dv, da),
        "time_gap": time_gap, "drac": drac(gap, dv),
    }


def exposure(ttc_values, dt, threshold=DEFAULT_TTC_THRESHOLD):
    """TET (s) and TIT (s^2) of TTC samples (one per pair and tick)."""
    ttc_values = np.asarray(ttc_values, float)
    threshold = np.broadcast_to(np.asarray(threshold, float), ttc_values.shape)
    hit = (ttc_values > 0) & (ttc_values <= threshold)
    weights = np.broadcast_to(np.asarray(dt, float), ttc_values.shape)
    tet = float(np.sum(weights[hit]))
    tit = float(np.sum((threshold - ttc_values)[hit] * weights[hit]))
    return tet, tit


# ------------------------------------------------------------------- PET
def _crossing_times(t, s, length, point, half_window):
    """Interpolated entry/exit time of a body passing ``point`` on its path."""
    front = s + length / 2.0
    rear = s - length / 2.0
    enter_at = point - half_window
    exit_at = point + half_window

    def first_cross(series, level):
        idx = np.nonzero(series >= level)[0]
        if len(idx) == 0:
            return None
        k = idx[0]
        if k == 0:
            return float(t[0]) if series[0] >= level else None
        s0, s1 = series[k - 1], series[k]
        frac = 0.0 if s1 == s0 else (level - s0) / (s1 - s0)
        return float(t[k - 1] + frac * (t[k] - t[k - 1]))

    entry = first_cross(front, enter_at)
    leave = first_cross(rear, exit_at)
    if entry is None or leave is None:
        return None
    return entry, leave


def _connector_paths(data):
    """Per vehicle that used a connector: (movement, t, s) along its path.

    ``s`` is measured from the connector start and stays continuous across
    the approach (negative) and the exit (beyond the connector length), so a
    body that straddles a seam is still tracked.
    """
    dt = float(data["dt"])
    names = data["road_names"]
    lengths = data["road_lengths"]
    kind = np.array([0 if str(n).endswith("_in") else 2 if str(n).endswith("_out") else 1
                     for n in names])
    road = data["road"]
    on = data["movement"] >= 0
    movement_of = {}
    conn_length = {}
    for row in np.nonzero(on)[0]:
        vid = int(data["vid"][row])
        if vid not in movement_of:
            movement_of[vid] = int(data["movement"][row])
            conn_length[vid] = float(lengths[road[row]])
    keep = np.isin(data["vid"], list(movement_of))
    rows = np.nonzero(keep)[0]
    rows = rows[np.lexsort((data["cycle"][rows], data["vid"][rows]))]
    vids = data["vid"][rows]
    breaks = np.nonzero(np.diff(vids))[0] + 1
    paths = {}
    for chunk in np.split(rows, breaks):
        if len(chunk) == 0:
            continue
        vid = int(data["vid"][chunk[0]])
        k = kind[road[chunk]]
        s = data["progress"][chunk].astype(float).copy()
        s[k == 0] -= lengths[road[chunk][k == 0]]
        s[k == 2] += conn_length[vid]
        paths.setdefault(movement_of[vid], []).append(
            (vid, data["cycle"][chunk] * dt, s, float(data["length"][chunk[0]])))
    return paths


def pet_events(data, conflict_points):
    """PET per pair of successive passages at each conflict point.

    ``conflict_points`` comes from :func:`trajectory.movement_conflict_points`.
    Returns arrays: pet (s; <= 0 means both bodies were in the conflict area
    at the same time), the two vehicle ids, movement codes and the time of
    the second entry.
    """
    paths = _connector_paths(data)
    out = {"pet": [], "vid_first": [], "vid_second": [], "movement_first": [],
           "movement_second": [], "time": []}
    for (ma, mb), points in conflict_points.items():
        if ma not in paths or mb not in paths:
            continue
        for sa, sb in points:
            times = []
            for m, s_point, other in ((ma, sa, mb), (mb, sb, ma)):
                # The conflict area along this path is as wide as the other
                # movement's occupant (car 1.7 m, motorbike 0.7 m).
                other_hw = 0.85 if _movement_is_car(data, other) else 0.35
                for vid, t, s, length in paths[m]:
                    crossing = _crossing_times(t, s, length, s_point, other_hw)
                    if crossing is not None:
                        times.append((crossing[0], crossing[1], vid, m))
            if len(times) < 2:
                continue
            times.sort()
            for first, second in zip(times, times[1:]):
                if first[3] == second[3]:
                    continue  # same movement: car-following, not a crossing
                out["pet"].append(second[0] - first[1])
                out["vid_first"].append(first[2])
                out["vid_second"].append(second[2])
                out["movement_first"].append(first[3])
                out["movement_second"].append(second[3])
                out["time"].append(second[0])
    return {key: np.asarray(value) for key, value in out.items()}


def _movement_is_car(data, code):
    name = str(data["movement_names"][code])
    return name.endswith(":2")


# -------------------------------------------------------------- 2-D TTC
def junction_ttc(data, horizon=5.0):
    """Disc TTC between vehicles on different connector movements per tick."""
    on = np.nonzero(data["movement"] >= 0)[0]
    cycles = data["cycle"][on]
    out = []
    radius = np.sqrt(data["length"] * 2 * data["half_width"] / math.pi)
    names = data["movement_names"]
    base = np.array([str(n).split("@")[0] for n in names])
    for cyc in np.unique(cycles):
        idx = on[cycles == cyc]
        if len(idx) < 2:
            continue
        p = np.stack([data["x"][idx], data["y"][idx]], axis=1)
        v = np.stack([np.cos(data["heading"][idx]), np.sin(data["heading"][idx])], axis=1) \
            * data["speed"][idx][:, None]
        r = radius[idx]
        i, j = np.triu_indices(len(idx), 1)
        different = base[data["movement"][idx[i]]] != base[data["movement"][idx[j]]]
        if not different.any():
            continue
        i, j = i[different], j[different]
        t = ttc_circle(p[i], v[i], r[i], p[j], v[j], r[j])
        keep = t <= horizon
        out.append(t[keep])
    return np.concatenate(out) if out else np.zeros(0)


# --------------------------------------------------------------- summary
def summarize(data, conflict_points=None, ttc_threshold=DEFAULT_TTC_THRESHOLD):
    dt = float(data["dt"])
    re = rear_end_pairs(data)
    finite = np.isfinite(re["mttc"])
    sample_dt = data.get("sample_dt")
    weights = (sample_dt[re["row_follower"]] if sample_dt is not None else dt)
    tet, tit = exposure(re["ttc"], weights, ttc_threshold)
    tet_m, tit_m = exposure(re["mttc"], weights, ttc_threshold)
    vehicle_seconds = (float(np.sum(sample_dt)) if sample_dt is not None else
                       len(data["vid"]) * dt)
    yellow_ticks = set(data["tick_cycle"][data["tick_yellow"] >= 0].tolist())
    in_yellow = np.isin(data["cycle"], list(yellow_ticks))
    hard = data["accel"] <= HARD_BRAKE
    summary = {
        "vehicle_hours": vehicle_seconds / 3600.0,
        "rear_end_samples": int(len(re["gap"])),
        "ttc_lt_1_5": int(np.sum(re["ttc"] < 1.5)),
        "mttc_lt_1_5": int(np.sum(re["mttc"] < 1.5)),
        "mttc_lt_1_0": int(np.sum(re["mttc"] < 1.0)),
        "time_gap_lt_0_5": int(np.sum(re["time_gap"] < 0.5)),
        "drac_gt_3_4": int(np.sum(re["drac"] > DRAC_COMFORT)),
        "drac_gt_5_0": int(np.sum(re["drac"] > DRAC_HARD)),
        "tet_ttc_s": tet, "tit_ttc_s2": tit,
        "tet_mttc_s": tet_m, "tit_mttc_s2": tit_m,
        "tit_mttc_per_veh_h": tit_m / max(vehicle_seconds / 3600.0, 1e-9),
        "body_overlap_samples": int(np.sum(re["gap"] < 0)),
        "hard_brake_samples": int(hard.sum()),
        "hard_brake_in_yellow": int((hard & in_yellow).sum()),
        "mttc_finite_share": float(finite.mean()) if len(finite) else 0.0,
    }
    if conflict_points is not None:
        pet = pet_events(data, conflict_points)
        values = pet["pet"]
        summary.update({
            "pet_events": int(len(values)),
            "pet_le_0": int(np.sum(values <= 0)),
            "pet_lt_1_0": int(np.sum(values < PET_CRITICAL)),
            "pet_lt_1_5": int(np.sum(values < PET_SERIOUS)),
        })
    jt = junction_ttc(data)
    summary.update({"junction_ttc_lt_1_5": int(np.sum(jt < 1.5)),
                    "junction_ttc_lt_1_0": int(np.sum(jt < 1.0))})
    return summary
