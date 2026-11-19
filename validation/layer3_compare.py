"""Layer 3: simulator vs AMIT field data, metric by metric.

    python3 validation/layer3_compare.py \
        --field validation/results/amit_field_key_metrics.csv \
        --sim amit_like=sim_runs/amit_like_summary.json hanoi_default=sim_runs/hanoi_default_summary.json \
        --match A01 --out validation/results/layer3_comparison

Both sides come from the same code (amit_events -> amit_metrics ->
amit_site_summary.key_metrics). For distributions (15th, 50th, 85th
percentiles) the distance is the mean absolute difference of the three
percentiles; the reference is the median of the same distance between pairs
of field sites. A metric passes when distance(sim, matched site) <= 1.5 x
reference. Gap acceptance is compared only with the two-phase sites A01 and
A03, where left turns are permissive.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amit_site_summary as S  # noqa: E402

TRIPLETS = [
    ("moto_v_stop_free", "Xe máy: tốc độ qua vạch dừng khi không phải dừng", "m/s"),
    ("car_v_stop_free", "Ô tô: tốc độ qua vạch dừng khi không phải dừng", "m/s"),
    ("moto_T_vint", "Xe máy đi thẳng: tốc độ trung bình trong nút", "m/s"),
    ("car_T_vint", "Ô tô đi thẳng: tốc độ trung bình trong nút", "m/s"),
    ("moto_L_vint", "Xe máy rẽ trái: tốc độ trung bình trong nút", "m/s"),
    ("car_L_vint", "Ô tô rẽ trái: tốc độ trung bình trong nút", "m/s"),
    ("moto_R_vint", "Xe máy rẽ phải: tốc độ trung bình trong nút", "m/s"),
    ("car_R_vint", "Ô tô rẽ phải: tốc độ trung bình trong nút", "m/s"),
    ("moto_L_alat", "Xe máy rẽ trái: gia tốc ngang lớn nhất", "m/s²"),
    ("car_L_alat", "Ô tô rẽ trái: gia tốc ngang lớn nhất", "m/s²"),
    ("moto_R_alat", "Xe máy rẽ phải: gia tốc ngang lớn nhất", "m/s²"),
    ("car_R_alat", "Ô tô rẽ phải: gia tốc ngang lớn nhất", "m/s²"),
    ("moto_alon_min", "Xe máy: giảm tốc mạnh nhất mỗi xe", "m/s²"),
    ("car_alon_min", "Ô tô: giảm tốc mạnh nhất mỗi xe", "m/s²"),
    ("moto_alon_max", "Xe máy: tăng tốc mạnh nhất mỗi xe", "m/s²"),
    ("car_alon_max", "Ô tô: tăng tốc mạnh nhất mỗi xe", "m/s²"),
    ("dis_sat_flow_per_m", "Lưu lượng xả bão hòa trên 1 m bề rộng", "xe/h/m"),
    ("dis_n_10s_per_m", "Số xe qua vạch trong 10 s đầu xanh, trên 1 m", "xe/m"),
]
SCALARS = [
    ("moto_share", "Tỷ lệ xe máy trong dòng xe", ""),
    ("moto_crossing_state_y", "Xe máy qua vạch khi vàng (tỷ lệ)", ""),
    ("moto_crossing_state_r", "Xe máy qua vạch khi đỏ (tỷ lệ)", ""),
    ("car_crossing_state_y", "Ô tô qua vạch khi vàng (tỷ lệ)", ""),
    ("car_crossing_state_r", "Ô tô qua vạch khi đỏ (tỷ lệ)", ""),
]
GAP_SITES = ("A01", "A03")
GAP_BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 12)]


def triplet(row, key):
    vals = [row.get(f"{key}_{q}") for q in ("p15", "p50", "p85")]
    if any(v is None or (isinstance(v, float) and not math.isfinite(v)) for v in vals):
        return None
    return np.array(vals, float)


def distance(a, b):
    return float(np.mean(np.abs(a - b))) if a is not None and b is not None else float("nan")


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, np.ndarray):
        return " / ".join(f"{v:.{nd}f}" for v in x)
    return f"{x:.{nd}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", required=True, type=Path)
    parser.add_argument("--sim", nargs="+", required=True, help="name=summary.json")
    parser.add_argument("--match", default="A01")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    field = pd.read_csv(args.field)
    field = {r["site"]: r for r in field.to_dict("records")}
    sims = {}
    for item in args.sim:
        name, path = item.split("=", 1)
        sims[name] = S.key_metrics(json.loads(Path(path).read_text()))
    names = list(sims)
    rows, lines = [], []
    head = (f"| Chỉ số | Đơn vị | {args.match} (p15/p50/p85) | Khoảng p50 6 nút | "
            + " | ".join(f"{n} (p15/p50/p85)" for n in names)
            + f" | Khoảng cách {names[0]}–{args.match} | Tham chiếu giữa các nút | Đạt? |")
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for key, label, unit in TRIPLETS:
        f = {s: triplet(r, key) for s, r in field.items()}
        f = {s: v for s, v in f.items() if v is not None}
        if args.match not in f:
            continue
        pair = [distance(f[a], f[b]) for a, b in itertools.combinations(sorted(f), 2)]
        ref = float(np.median(pair)) if pair else float("nan")
        p50 = [v[1] for v in f.values()]
        s_vals = {n: triplet(sims[n], key) for n in names}
        d = distance(s_vals[names[0]], f[args.match])
        ok = d <= 1.5 * ref if math.isfinite(d) and math.isfinite(ref) else None
        rows.append({"metric": key, "label": label, "unit": unit, "field_match": fmt(f[args.match]),
                     "field_p50_min": min(p50), "field_p50_max": max(p50),
                     **{f"sim_{n}": fmt(s_vals[n]) for n in names},
                     "distance": d, "reference": ref, "pass": ok})
        lines.append(f"| {label} | {unit} | {fmt(f[args.match])} | {min(p50):.2f}–{max(p50):.2f} | "
                     + " | ".join(fmt(s_vals[n]) for n in names)
                     + f" | {fmt(d)} | {fmt(ref)} | {'đạt' if ok else ('không' if ok is False else '–')} |")
    lines += ["", f"| Chỉ số | {args.match} | Khoảng 6 nút | " + " | ".join(names) + " | Trong khoảng? |",
              "|---|---|---|" + "---|" * len(names) + "---|"]
    for key, label, unit in SCALARS:
        vals = [r.get(key) for r in field.values() if r.get(key) is not None and math.isfinite(r.get(key))]
        sv = [sims[n].get(key) for n in names]
        inside = vals and sv[0] is not None and min(vals) <= sv[0] <= max(vals)
        rows.append({"metric": key, "label": label, "field_match": field[args.match].get(key),
                     "field_min": min(vals), "field_max": max(vals),
                     **{f"sim_{n}": v for n, v in zip(names, sv)}, "pass": bool(inside)})
        lines.append(f"| {label} | {fmt(field[args.match].get(key), 3)} | {min(vals):.3f}–{max(vals):.3f} | "
                     + " | ".join(fmt(v, 3) for v in sv) + f" | {'có' if inside else 'không'} |")
    # gap acceptance
    lines += ["", "Chấp nhận khe rẽ trái (khe giữa hai xe đi thẳng đối diện tại điểm xung đột; "
              "chỉ các nút 2 pha):", "",
              "| Khe (s) | " + " | ".join(f"{s} p (n)" for s in GAP_SITES) + " | "
              + " | ".join(f"{n} p (n)" for n in names) + f" | z ({names[0]} vs gộp {'+'.join(GAP_SITES)}) |",
              "|---|" + "---|" * (len(GAP_SITES) + len(names) + 1)]
    for lo, hi in GAP_BINS:
        cells, pooled_k, pooled_n = [], 0.0, 0.0
        for s in GAP_SITES:
            p, n = field[s].get(f"gap_p_{lo}_{hi}"), field[s].get(f"gap_n_{lo}_{hi}")
            if p is not None and n is not None and math.isfinite(n):
                pooled_k += p * n
                pooled_n += n
            cells.append(f"{fmt(p)} ({int(n) if n is not None and math.isfinite(n) else 0})")
        sim_cells = []
        z = float("nan")
        for i, nme in enumerate(names):
            p, n = sims[nme].get(f"gap_p_{lo}_{hi}"), sims[nme].get(f"gap_n_{lo}_{hi}")
            sim_cells.append(f"{fmt(p)} ({int(n) if n else 0})")
            if i == 0 and p is not None and n and pooled_n:
                pf = pooled_k / pooled_n
                se = math.sqrt(max(pf * (1 - pf), 1e-4) / pooled_n + max(p * (1 - p), 1e-4) / n)
                z = (p - pf) / se
        rows.append({"metric": f"gap_p_{lo}_{hi}", "z": z, "pass": abs(z) <= 2 if math.isfinite(z) else None})
        lines.append(f"| {lo}–{hi} | " + " | ".join(cells) + " | " + " | ".join(sim_cells) + f" | {fmt(z)} |")
    lines += ["", "| Khe tới hạn (logit, s) | " + " | ".join(GAP_SITES) + " | " + " | ".join(names) + " |",
              "|---|" + "---|" * (len(GAP_SITES) + len(names))]
    for who in ("all", "moto", "car"):
        lines.append(f"| {who} | " + " | ".join(fmt(field[s].get(f'gap_tc_{who}')) for s in GAP_SITES)
                     + " | " + " | ".join(fmt(sims[n].get(f'gap_tc_{who}')) for n in names) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out.with_suffix(".csv"), index=False)
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
