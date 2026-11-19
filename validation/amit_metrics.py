"""Validation metrics, computed the same way on field and simulated event tables.

Input: the tables of :mod:'amit_events' (one row per vehicle, conflict-point
passages) and the recording's signal table. Output: plain dicts / DataFrames
that can be pooled over recordings and compared between field and simulation.
"""
import math

import numpy as np
import pandas as pd

QUANTILES = np.round(np.linspace(0.05, 0.95, 19), 2)
FIT_START = 10.0   # s after green onset, start of the saturation window (as calibration.py)


def green_intervals(sig, column):
    """[(green start, green end, yellow end)] for one movement column."""
    out, start, g_end = [], None, None
    states = sig[column].astype(str).str.strip().to_numpy() if column in sig.columns else []
    for state, t0, t1 in zip(states, sig["Timestep"].to_numpy(), sig["End"].to_numpy()):
        if state == "g":
            if start is None:
                start = t0
            g_end = t1
        elif state == "y" and start is not None:
            out.append((float(start), float(g_end), float(t1)))
            start = None
        elif start is not None:
            out.append((float(start), float(g_end), float(g_end)))
            start = None
    return out


def effective_width(ev, arm):
    """Width used by the traffic of one approach at its stop line (m).

    Spread (1st to 99th percentile) of the lateral position of vehicle centres
    where they cross the stop line, plus the median vehicle width. Drawn stop
    lines differ between recordings of the same site, so flows are normalised
    by this width instead.
    """
    sub = ev[(ev["inn"] == arm) & ev["lat_stop"].notna()]
    if len(sub) < 20:
        return np.nan
    lat = sub["lat_stop"].to_numpy()
    return float(np.quantile(lat, 0.99) - np.quantile(lat, 0.01) + sub["width"].median())


def discharge(ev, sig, arm, width, column=None):
    """Saturated discharge of one approach, per green with a standing queue.

    A green counts when at least five vehicles stood (v < 0.5 m/s for 1 s)
    within 80 m before its onset and the last of them crosses the discharge
    line (stop line, or front of the motorcycle waiting area) at least 15 s
    after onset. Saturation flow is the slope of the cumulative count at that
    line (all movements of the approach) from FIT_START to the crossing of the
    last queued vehicle; lost time = FIT_START - N(FIT_START)/s.
    """
    rows = []
    column = column or arm + "T"
    sub_arm = ev[(ev["inn"] == arm) & ev["t_dis"].notna()]
    for g0, g1, y1 in green_intervals(sig, column):
        sub = sub_arm[(sub_arm["t_dis"] >= g0) & (sub_arm["t_dis"] <= y1)]
        queued = sub[sub["t_queue"] < g0]
        if len(queued) < 5:
            continue
        t_last = float(queued["t_dis"].max() - g0)
        if t_last < FIT_START + 5:
            continue
        times = np.sort(sub["t_dis"].to_numpy() - g0)
        window = times[(times >= FIT_START) & (times <= t_last)]
        if len(window) < 5:
            continue
        counts = np.searchsorted(times, window, side="right")
        slope = float(np.polyfit(window, counts, 1)[0])
        if slope <= 0:
            continue
        before = int(np.sum(times < FIT_START))
        rows.append({"arm": arm, "g0": g0, "green": g1 - g0, "n_queued": len(queued),
                     "t_last_queued": t_last, "sat_flow": slope * 3600.0,
                     "sat_flow_per_m": slope * 3600.0 / width,
                     "lost_time": FIT_START - before / slope,
                     "n_10s_per_m": float(np.sum(times <= 10.0)) / width,
                     "n_20s_per_m": float(np.sum(times <= 20.0)) / width,
                     "moto_share_queued": float((queued["kind"] == "moto").mean())})
    return pd.DataFrame(rows)


def gaps(passages, ev, sig):
    """Gaps between successive opposing through vehicles at each left-turn conflict point.

    For every left-turner, the gaps that start after it crossed its discharge line
    and end before it passes the conflict point are rejected; the gap in which
    it passes is accepted. Lags are left out (their start depends on where the
    turner waits). Only gaps that start while the opposing through movement
    shows green are kept, i.e. permissive left turns.
    """
    rows = []
    if passages is None or len(passages) == 0:
        return pd.DataFrame(rows)
    t_stop = ev.set_index("vid")["t_dis"]
    for cp, grp in passages.groupby("cp"):
        opp = cp.split("-")[1][0]
        thr = np.sort(grp[grp["role"] == "through"]["t"].to_numpy())
        if len(thr) < 2:
            continue
        for r in grp[grp["role"] == "left"].itertuples():
            t_arr = t_stop.get(r.vid, np.nan)
            if not np.isfinite(t_arr) or r.t <= t_arr:
                continue
            k0 = int(np.searchsorted(thr, t_arr))
            k1 = int(np.searchsorted(thr, r.t))
            if k1 >= len(thr):
                continue
            for k in range(k0, k1):
                if k == 0:
                    continue
                start, end = thr[k - 1], thr[k]
                if start < t_arr:
                    continue
                accepted = end > r.t
                if signal_green(sig, opp + "T", start):
                    rows.append({"cp": cp, "vid": r.vid, "kind": r.kind, "gap": end - start,
                                 "accepted": int(accepted)})
            if k1 >= 1 and thr[k1 - 1] >= t_arr and signal_green(sig, opp + "T", thr[k1 - 1]):
                rows.append({"cp": cp, "vid": r.vid, "kind": r.kind,
                             "gap": thr[k1] - thr[k1 - 1], "accepted": 1})
    return pd.DataFrame(rows).drop_duplicates()


def signal_green(sig, column, t):
    if column not in sig.columns:
        return False
    i = int(np.searchsorted(sig["Timestep"].to_numpy(), t, side="right")) - 1
    return i >= 0 and str(sig[column].iloc[i]).strip() == "g"


def logit_fit(x, y, iters=50):
    """Binary logit P(y=1) = 1 / (1 + exp(-(b0 + b1 x))) by Newton-Raphson."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or y.min() == y.max():
        return {"b0": np.nan, "b1": np.nan, "critical_gap": np.nan, "n": int(len(x))}
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    ll = float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
    return {"b0": float(b[0]), "b1": float(b[1]),
            "critical_gap": float(-b[0] / b[1]) if b[1] != 0 else np.nan,
            "loglik": ll, "n": int(len(x))}


def acceptance_curve(gap_table, bins=(0, 1, 2, 3, 4, 5, 6, 8, 12)):
    out = []
    if len(gap_table) == 0:
        return out
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = gap_table[(gap_table["gap"] >= lo) & (gap_table["gap"] < hi)]
        if len(sel):
            p = float(sel["accepted"].mean())
            se = math.sqrt(max(p * (1 - p), 1e-9) / len(sel))
            out.append({"lo": lo, "hi": hi, "n": int(len(sel)), "p": p, "se": se})
    return out


def quantiles(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return None
    return [float(q) for q in np.quantile(v, QUANTILES)]


def vehicle_table_metrics(ev):
    """Distributions over vehicles: speeds, turning, travel times, accelerations, compliance."""
    m = {}
    veh = ev[ev["inn"] != "X"]
    m["n_vehicles"] = int(len(ev))
    m["share"] = {k: float(v) for k, v in ev["kind"].value_counts(normalize=True).items()}
    m["moto_share_motorised"] = float((ev["kind"] == "moto").mean())
    for kind in ("moto", "car"):
        k = veh[veh["kind"] == kind]
        free = k[(k["mov"] == "T") & k["t_queue"].isna() & (k["sig_dis"] == "g")]
        m[f"{kind}_v_stop_free"] = quantiles(free["v_stop"])
        for mov in ("T", "L", "R"):
            km = k[k["mov"] == mov]
            m[f"{kind}_{mov}_tt"] = quantiles(km["t_exit"] - km["t_stop"])
            m[f"{kind}_{mov}_alat"] = quantiles(km["a_lat_max"])
            m[f"{kind}_{mov}_vint"] = quantiles(km["v_mean_int"])
        m[f"{kind}_alon_min"] = quantiles(k["a_lon_min"])
        m[f"{kind}_alon_max"] = quantiles(k["a_lon_max"])
        m[f"{kind}_u_stop_T"] = quantiles(k[k["mov"] == "T"]["u_stop"])
        cross = k[k["t_dis"].notna() & (k["mov"] != "R")]
        if len(cross):
            m[f"{kind}_crossing_state"] = {s: float(v) for s, v in
                                           cross["sig_dis"].value_counts(normalize=True).items()}
    return m


def flows(ev, bin_s=900.0, t_start=0.0):
    """Stop-line counts per (approach, movement) and time bin."""
    sub = ev[ev["t_dis"].notna() & (ev["inn"] != "X")].copy()
    sub["bin"] = ((sub["t_dis"] - t_start) // bin_s).astype(int)
    return sub.groupby(["bin", "inn", "mov"]).size().rename("n").reset_index()


def geh(model, observed):
    """GEH on hourly flows."""
    m, c = np.asarray(model, float), np.asarray(observed, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m + c > 0, np.sqrt(2 * (m - c) ** 2 / (m + c)), 0.0)


def theil_u(sim, obs):
    s, o = np.asarray(sim, float), np.asarray(obs, float)
    num = math.sqrt(np.mean((s - o) ** 2))
    den = math.sqrt(np.mean(s ** 2)) + math.sqrt(np.mean(o ** 2))
    return num / den if den > 0 else np.nan


def wasserstein_q(qa, qb):
    """L1 distance between two distributions given by the same quantile grid."""
    if qa is None or qb is None:
        return np.nan
    return float(np.mean(np.abs(np.asarray(qa) - np.asarray(qb))))
