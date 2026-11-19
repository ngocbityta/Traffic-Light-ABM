"""Summarise AMIT recordings (field or simulated) into validation metrics.

    python3 validation/amit_site_summary.py data/amit/A01 --out validation/results/A01_field.json

Reads every '<rec>_TRJ.csv' with its '_linemarking.csv' and '_signal.csv' in
a site folder, runs amit_events + amit_metrics and writes one JSON summary.
The same function runs in the browser (Pyodide) on the published zip files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amit_events as ae  # noqa: E402
import amit_metrics as am  # noqa: E402


def summarise_recording(rec, trj_text, lm_text, sig_text):
    lines = ae.read_linemarking(lm_text)
    sig = ae.read_signal(sig_text)
    trj = ae.load_trj(trj_text)
    ev, tracks = ae.extract(trj, lines, sig, rec)
    cps = ae.conflict_points(ev, tracks)
    passages = ae.conflict_passages(ev, tracks, cps)
    geo = ev.attrs["geo"]
    ev.attrs = {}  # numpy arrays in attrs break pd.concat
    widths = {arm: am.effective_width(ev, arm) for arm in geo}
    dis = [am.discharge(ev, sig, arm, widths[arm]) for arm in geo if np.isfinite(widths[arm])]
    dis = pd.concat([d for d in dis if len(d)], ignore_index=True) if any(len(d) for d in dis) else pd.DataFrame()
    if len(dis):
        dis["rec"] = rec
    gp = am.gaps(passages, ev, sig)
    if len(gp):
        gp["rec"] = rec
    fl = am.flows(ev)
    fl["rec"] = rec
    info = {"rec": rec, "duration_s": float(trj["t"].max()), "n_tracks": int(len(ev)),
            "widths": {a: round(g["width"], 2) for a, g in geo.items()},
            "effective_widths": {a: round(w, 2) for a, w in widths.items()},
            "box_depth": {a: round(g["box"], 2) for a, g in geo.items()},
            "conflict_points": {k: [round(float(c), 2) for c in v[2]] for k, v in cps.items()},
            "n_passages": int(len(passages))}
    return ev, dis, gp, fl, info


def summarise_site(site, recordings):
    """recordings: iterable of (rec, trj_text, lm_text, sig_text)."""
    return aggregate(site, [summarise_recording(*r) for r in recordings])


def aggregate(site, results):
    """Pool the outputs of summarise_recording over the recordings of a site."""
    evs, diss, gps, fls, infos = (list(x) for x in zip(*results))
    ev = pd.concat(evs, ignore_index=True)
    dis = pd.concat([d for d in diss if len(d)], ignore_index=True) if any(len(d) for d in diss) else pd.DataFrame()
    gp = pd.concat([g for g in gps if len(g)], ignore_index=True) if any(len(g) for g in gps) else pd.DataFrame()
    out = {"site": site, "recordings": infos, "vehicles": am.vehicle_table_metrics(ev)}
    if len(dis):
        out["discharge"] = {
            "n_greens": int(len(dis)),
            "sat_flow_per_m": am.quantiles(dis["sat_flow_per_m"]),
            "lost_time": am.quantiles(dis["lost_time"]),
            "n_10s_per_m": am.quantiles(dis["n_10s_per_m"]),
            "n_20s_per_m": am.quantiles(dis["n_20s_per_m"]),
            "by_recording": dis.groupby("rec")[["sat_flow_per_m", "lost_time"]].median().round(3).reset_index().to_dict("records"),
            "by_arm": dis.groupby("arm")[["sat_flow_per_m", "lost_time"]].median().round(3).reset_index().to_dict("records"),
            "median": {"sat_flow_per_m": float(dis["sat_flow_per_m"].median()),
                       "lost_time": float(dis["lost_time"].median()),
                       "moto_share_queued": float(dis["moto_share_queued"].median())},
        }
    if len(gp):
        out["gaps"] = {"all": am.logit_fit(gp["gap"], gp["accepted"]),
                       "curve": am.acceptance_curve(gp)}
        for kind in ("moto", "car"):
            k = gp[gp["kind"] == kind]
            out["gaps"][kind] = am.logit_fit(k["gap"], k["accepted"])
            out["gaps"][kind + "_curve"] = am.acceptance_curve(k)
        out["gaps"]["by_recording"] = {rec: am.logit_fit(g["gap"], g["accepted"])["critical_gap"]
                                       for rec, g in gp.groupby("rec")}
    return out, ev, dis, gp


KEY_QUANTILES = {"p15": 2, "p50": 8, "p85": 15}   # positions in amit_metrics.QUANTILES


def key_metrics(out):
    """Flat dict of the headline numbers of one site summary (for tables)."""
    k = {"site": out["site"]}
    veh = out["vehicles"]
    k["moto_share"] = veh["moto_share_motorised"]
    for name, q in veh.items():
        if isinstance(q, list) and len(q) == len(am.QUANTILES):
            for label, i in KEY_QUANTILES.items():
                k[f"{name}_{label}"] = q[i]
        elif name.endswith("crossing_state"):
            for state in ("y", "r"):
                k[f"{name}_{state}"] = q.get(state, 0.0)
    dis = out.get("discharge")
    if dis:
        k["n_greens"] = dis["n_greens"]
        for name in ("sat_flow_per_m", "lost_time", "n_10s_per_m", "n_20s_per_m"):
            for label, i in KEY_QUANTILES.items():
                k[f"dis_{name}_{label}"] = dis[name][i] if dis[name] else None
    gaps = out.get("gaps")
    if gaps:
        for who in ("all", "moto", "car"):
            k[f"gap_tc_{who}"] = gaps[who]["critical_gap"]
            k[f"gap_n_{who}"] = gaps[who]["n"]
        for b in gaps["curve"]:
            k[f"gap_p_{b['lo']}_{b['hi']}"] = b["p"]
            k[f"gap_n_{b['lo']}_{b['hi']}"] = b["n"]
    return k


def _site_files(folder):
    folder = Path(folder)
    for trj in sorted(folder.glob("*_TRJ.csv")):
        rec = trj.name[:-len("_TRJ.csv")]
        yield (rec, trj.read_text(), (folder / f"{rec}_linemarking.csv").read_text(),
               (folder / f"{rec}_signal.csv").read_text())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    site = Path(args.folder).name
    out, ev, dis, gp = summarise_site(site, _site_files(args.folder))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    stem = Path(args.out).with_suffix("")
    ev.to_csv(f"{stem}_events.csv", index=False)
    if len(dis):
        dis.to_csv(f"{stem}_discharge.csv", index=False)
    if len(gp):
        gp.to_csv(f"{stem}_gaps.csv", index=False)


if __name__ == "__main__":
    main()
