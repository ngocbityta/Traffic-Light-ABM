"""Per-vehicle events from one recording in AMIT format (field data or simulator).

Field data and simulator output go through the same code: the simulator is
first exported to AMIT files ('_TRJ.csv', '_linemarking.csv',
'_signal.csv'), then :func:'extract' gives one row per track with stop-line
and exit-line crossings, queueing, turning kinematics and entry state, and
:func:'conflict_passages' gives the times at which left-turners and opposing
through vehicles pass their conflict point.

AMIT conventions (Data Format v1.3): metres in the frame of the background
image (Y down), 10 Hz, approaches A-D clockwise, entry label 'XI' / exit label
'YO' set by crossing the stop line / exit line. Movement from the labels:
exit one letter after entry = left, two = through, three = right, same = U.
"""
import io
import math

import numpy as np
import pandas as pd

ORDER = "ABCD"
KIND = {"c": "car", "m": "moto", "b": "heavy", "t": "heavy", "g": "heavy", "h": "heavy"}
MOVES = {0: "U", 1: "L", 2: "T", 3: "R"}
SIGNAL_OF = {"L": "L", "U": "L", "T": "T", "R": "R"}
STOP_SPEED = 0.5      # m/s, below = standing
STOP_MIN_S = 1.0      # s standing to count as queued
QUEUE_REACH = 80.0    # m upstream of the stop line
CP_RADIUS = 3.0       # m, max distance of a track to a conflict point
BOX_MAX = 10.0        # m, deepest motorcycle waiting area accepted


def read_linemarking(text):
    lines = {"stop": {}, "exit": {}}
    block = None
    for raw in text.splitlines():
        cells = [c.strip() for c in raw.split(",")]
        head = cells[0].lower()
        if head in ("stopline", "exitline"):
            block = head[:4]
            continue
        if block and cells[0] in ORDER and len(cells) >= 5 and cells[1]:
            lines[block][cells[0]] = np.array(list(map(float, cells[1:5]))).reshape(2, 2)
    return lines


def read_signal(text):
    sig = pd.read_csv(io.StringIO(text))
    sig.columns = [c.strip() for c in sig.columns]
    sig = sig[sig["Duration"] > 0].reset_index(drop=True)
    sig["End"] = sig["Timestep"] + sig["Duration"]
    return sig


def signal_state(sig, column, t):
    """State ('g', 'y', 'r', '-') of one movement column at time t."""
    if column not in sig.columns or not np.isfinite(t):
        return "-"
    i = int(np.searchsorted(sig["Timestep"].to_numpy(), t, side="right")) - 1
    if i < 0:
        return "-"
    return str(sig[column].iloc[i]).strip()


def signal_states(sig, column, times):
    """Vectorised :func:'signal_state'."""
    times = np.asarray(times, float)
    if column not in sig.columns:
        return np.full(len(times), "-", dtype=object)
    idx = np.searchsorted(sig["Timestep"].to_numpy(), times, side="right") - 1
    states = sig[column].astype(str).str.strip().to_numpy()
    return np.where(idx >= 0, states[np.clip(idx, 0, None)], "-")


def green_onsets(sig, column):
    """Times at which one movement column turns green."""
    if column not in sig.columns:
        return np.array([])
    states = sig[column].astype(str).str.strip().to_numpy()
    starts = sig["Timestep"].to_numpy()
    prev = np.concatenate([["-"], states[:-1]])
    return starts[(states == "g") & (prev != "g")]


def geometry(lines):
    """Centre, outward axis per approach, stop lines oriented centre side -> curb."""
    pts = np.vstack([seg for block in lines.values() for seg in block.values()])
    centre = pts.mean(axis=0)
    geo = {}
    for arm, seg in lines["stop"].items():
        mid = seg.mean(axis=0)
        axis = mid - centre
        axis = axis / np.hypot(*axis)
        ex = lines["exit"].get(arm)
        if ex is not None:
            ex_mid = ex.mean(axis=0)
            d0, d1 = np.hypot(*(seg[0] - ex_mid)), np.hypot(*(seg[1] - ex_mid))
            if d1 < d0:
                seg = seg[::-1]
        geo[arm] = {"stop": seg, "axis": axis, "s_stop": float(np.dot(mid - centre, axis)),
                    "width": float(np.hypot(*(seg[1] - seg[0])))}
    return centre, geo


def _crossing(P, seg, ext=1.0):
    """First crossing of polyline P with segment 'seg' (extended by 'ext').

    Returns (sample index i, fraction within [i, i+1], position along the
    original segment as a fraction of its length) or None.
    """
    p, q = seg[0], seg[1]
    d = q - p
    length = float(np.hypot(*d))
    if length <= 0 or len(P) < 2:
        return None
    d = d / length
    p, q = p - d * ext, q + d * ext
    a, r = P[:-1], P[1:] - P[:-1]
    s = q - p
    den = r[:, 0] * s[1] - r[:, 1] * s[0]
    qp = p - a
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (qp[:, 0] * s[1] - qp[:, 1] * s[0]) / den
        uu = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / den
    ok = (den != 0) & (tt >= 0) & (tt <= 1) & (uu >= 0) & (uu <= 1)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None
    i = int(idx[0])
    along = (uu[i] * (length + 2 * ext) - ext) / length
    return i, float(tt[i]), float(along)


def _smooth(values, window=5):
    if len(values) < window:
        return values.astype(float)
    kernel = np.ones(window) / window
    padded = np.pad(values.astype(float), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _kinematics(t, P, v):
    """Lateral acceleration (|v * yaw rate|) and longitudinal acceleration, 0.5 s stencils."""
    n = len(t)
    a_lat = np.full(n, np.nan)
    a_lon = np.full(n, np.nan)
    if n < 11:
        return a_lat, a_lon
    x, y = _smooth(P[:, 0]), _smooth(P[:, 1])
    k = 5
    dx, dy = x[2 * k:] - x[:-2 * k], y[2 * k:] - y[:-2 * k]
    moving = np.hypot(dx, dy) > 0.5
    head = np.unwrap(np.arctan2(dy, dx))
    head_full = np.full(n, np.nan)
    head_full[k:n - k] = np.where(moving, head, np.nan)
    dt2 = t[2 * k:] - t[:-2 * k]
    yaw = (head_full[2 * k:] - head_full[:-2 * k]) / dt2
    a_lat[2 * k:] = np.abs(v[k:n - k] * yaw)
    a_lat = np.roll(a_lat, -k)
    a_lat[n - k:] = np.nan
    vs = _smooth(v)
    a_lon[k:n - k] = (vs[2 * k:] - vs[:-2 * k]) / dt2
    return a_lat, a_lon


def load_trj(text):
    trj = pd.read_csv(io.StringIO(text))
    trj.columns = [c.strip() for c in trj.columns]
    trj["cx"] = (trj["Front X"] + trj["Rear X"]) / 2
    trj["cy"] = (trj["Front Y"] + trj["Rear Y"]) / 2
    trj["t"] = trj["Frame"] * 0.1
    return trj.sort_values(["Vehicle ID", "Frame"], kind="stable")


def extract(trj, lines, sig, rec=""):
    """One row per vehicle track (pedestrians and bicycles excluded).

    Besides the stop line, each approach gets a discharge line: the stop line
    moved towards the centre by the depth of the motorcycle waiting area, if
    any: most downstream 5 % of the positions of through / right-turn
    vehicles that stand beyond the stop line when their own green starts
    (at least 20 such vehicles, capped at BOX_MAX m). Queueing and discharge are timed at this line, so a waiting
    area in front of the stop line does not count as red-light running and
    the vehicles standing in it count as queued.
    """
    centre, geo = geometry(lines)
    veh = trj[trj["class"].isin(list(KIND))]
    raw = []
    box = {arm: [] for arm in geo}
    onsets = {arm: green_onsets(sig, arm + "T") for arm in geo}
    for vid, g in veh.groupby("Vehicle ID", sort=False):
        t = g["t"].to_numpy()
        P = g[["cx", "cy"]].to_numpy()
        v = g["Speed"].to_numpy(dtype=float)
        cls = str(g["class"].iloc[0])
        inn = str(g["intersection in"].iloc[0]).strip()
        out = str(g["intersection out"].iloc[0]).strip()
        arm_in = inn[0] if inn.endswith("I") and inn[0] in ORDER else None
        arm_out = out[0] if out.endswith("O") and out[0] in ORDER else None
        mov = "X"
        if arm_in and arm_out:
            mov = MOVES[(ORDER.index(arm_out) - ORDER.index(arm_in)) % 4]
        s_up = None
        if arm_in and arm_in in geo:
            G = geo[arm_in]
            s_up = (P - centre) @ G["axis"] - G["s_stop"]
            if mov in ("T", "R"):
                # standing beyond the stop line at the moment its own green starts
                for g0 in onsets[arm_in]:
                    j = int(np.searchsorted(t, g0)) - 1
                    if 0 <= j < len(t) - 1 and g0 - t[j] < 0.2 and v[j] < STOP_SPEED \
                            and -BOX_MAX - 5 < s_up[j] < -0.5:
                        box[arm_in].append(float(s_up[j]))
                        break
        raw.append((vid, g, t, P, v, cls, arm_in, arm_out, mov, s_up))
    for arm, values in box.items():
        depth = -float(np.quantile(values, 0.05)) if len(values) >= 20 else 0.0
        geo[arm]["box"] = float(min(max(depth, 0.0), BOX_MAX))
        seg = geo[arm]["stop"] - geo[arm]["axis"] * geo[arm]["box"]
        geo[arm]["discharge"] = seg
    rows, tracks = [], {}
    for vid, g, t, P, v, cls, arm_in, arm_out, mov, s_up in raw:
        row = {"rec": rec, "vid": int(vid), "cls": cls, "kind": KIND[cls],
               "length": float(g["Length"].iloc[0]), "width": float(g["Width"].iloc[0]),
               "inn": arm_in or "X", "out": arm_out or "X", "mov": mov,
               "t_first": float(t[0]), "t_last": float(t[-1]), "v_first": float(v[0])}
        a_lat, a_lon = _kinematics(t, P, v)
        row["a_lon_min"] = float(np.nanmin(a_lon)) if np.isfinite(a_lon).any() else np.nan
        row["a_lon_max"] = float(np.nanmax(a_lon)) if np.isfinite(a_lon).any() else np.nan
        i_stop = i_exit = None
        if s_up is not None:
            G = geo[arm_in]
            lat = (P - G["stop"][0]) @ ((G["stop"][1] - G["stop"][0]) / G["width"]) / G["width"]
            row["d_first"], row["u_first"] = float(s_up[0]), float(lat[0])
            c = _crossing(P, G["stop"])
            if c is not None:
                i, f, along = c
                i_stop = i
                row["t_stop"] = float(t[i] + f * (t[i + 1] - t[i]))
                row["v_stop"] = float(v[i] + f * (v[i + 1] - v[i]))
                row["u_stop"] = along
                row["lat_stop"] = along * G["width"]
            col = arm_in + SIGNAL_OF.get(mov, "T")
            c = _crossing(P, G["discharge"])
            i_dis = None
            if c is not None:
                i, f, _ = c
                i_dis = i
                row["t_dis"] = float(t[i] + f * (t[i + 1] - t[i]))
                row["v_dis"] = float(v[i] + f * (v[i + 1] - v[i]))
                row["sig_dis"] = signal_state(sig, col, row["t_dis"])
            end = i_dis if i_dis is not None else len(t) - 1
            reach = s_up[:end + 1]
            standing = (v[:end + 1] < STOP_SPEED) & (reach > -G["box"] - 1.0) & (reach < QUEUE_REACH)
            need = int(round(STOP_MIN_S / 0.1))
            run = 0
            for j, flag in enumerate(standing):
                run = run + 1 if flag else 0
                if run >= need:
                    j0 = j - need + 1
                    row["t_queue"] = float(t[j0])
                    row["d_queue"] = float(reach[j0])
                    break
        if arm_out and arm_out in lines["exit"]:
            c = _crossing(P, lines["exit"][arm_out])
            if c is not None:
                i, f, _ = c
                i_exit = i
                row["t_exit"] = float(t[i] + f * (t[i + 1] - t[i]))
                row["v_exit"] = float(v[i] + f * (v[i + 1] - v[i]))
        if i_stop is not None and i_exit is not None and i_exit > i_stop:
            seg = slice(i_stop, i_exit + 2)
            al = a_lat[seg]
            row["a_lat_max"] = float(np.nanmax(al)) if np.isfinite(al).any() else np.nan
            row["v_mean_int"] = float(np.mean(v[seg]))
            row["path_int"] = float(np.sum(np.hypot(*np.diff(P[seg], axis=0).T)))
            tracks[int(vid)] = (t[seg], P[seg])
        rows.append(row)
    ev = pd.DataFrame(rows)
    for col in ("t_stop", "v_stop", "u_stop", "lat_stop", "t_dis", "v_dis", "sig_dis", "t_queue", "d_queue",
                "t_exit", "v_exit", "a_lat_max", "v_mean_int", "path_int", "d_first", "u_first"):
        if col not in ev.columns:
            ev[col] = np.nan
    ev.attrs["centre"] = centre
    ev.attrs["geo"] = geo
    return ev, tracks


def _resample(P, n=40):
    seglen = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    if s[-1] <= 0:
        return None
    grid = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(grid, s, P[:, 0]), np.interp(grid, s, P[:, 1])])


def median_path(ev, tracks, arm_in, mov, n=40, min_tracks=10):
    vids = ev[(ev["inn"] == arm_in) & (ev["mov"] == mov)]["vid"]
    paths = [_resample(tracks[v][1], n) for v in vids if v in tracks and len(tracks[v][1]) > 3]
    paths = [p for p in paths if p is not None]
    if len(paths) < min_tracks:
        return None
    return np.median(np.stack(paths), axis=0)


def _polyline_intersection(A, B):
    for i in range(len(A) - 1):
        c = _crossing(B, A[i:i + 2], ext=0.0)
        if c is not None:
            j, f, _ = c
            return B[j] + f * (B[j + 1] - B[j])
    return None


def conflict_points(ev, tracks):
    """Left turn from X against through from the opposite approach (X+2)."""
    cps = {}
    for k, arm in enumerate(ORDER):
        opp = ORDER[(k + 2) % 4]
        left = median_path(ev, tracks, arm, "L")
        thr = median_path(ev, tracks, opp, "T")
        if left is None or thr is None:
            continue
        p = _polyline_intersection(left, thr)
        if p is not None:
            cps[f"{arm}L-{opp}T"] = (arm, opp, p)
    return cps


def conflict_passages(ev, tracks, cps):
    rows = []
    kinds = ev.set_index("vid")["kind"].to_dict()
    for name, (arm, opp, p) in cps.items():
        for role, a, mov in (("left", arm, "L"), ("through", opp, "T")):
            vids = ev[(ev["inn"] == a) & (ev["mov"] == mov)]["vid"]
            for v in vids:
                if v not in tracks:
                    continue
                t, P = tracks[v]
                d = np.hypot(*(P - p).T)
                i = int(np.argmin(d))
                if d[i] <= CP_RADIUS:
                    rows.append({"cp": name, "vid": int(v), "role": role, "kind": kinds[v],
                                 "t": float(t[i]), "dist": float(d[i])})
    return pd.DataFrame(rows, columns=["cp", "vid", "role", "kind", "t", "dist"])
