"""Kiểm tra hàm generate_demand: thông số, seed data, tính tất định.

    uv run --no-project --python 3.12 --with "numpy<2" python validation/check_demand.py
    uv run --no-project --python 3.12 --with "numpy<2" python validation/check_demand.py --negative-control

Hai hàm được kiểm:
  python  python_simulation.demand.generate_demand   (train LVL dùng, mã rẽ 0=trái 1=phải 2=thẳng)
  gama    train.logic.demand.generate_demand(site="lvl")   (benchmark GAMA, mã rẽ 0=thẳng 1=trái 2=phải)

Ghi validation/results/demand_check_report.md và .json.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python-simulation" / "simulation"))

from python_simulation import demand as pydemand  # noqa: E402
from python_simulation.python_sim import (ARMS, DEFAULT_HEAVY_RATIO,  # noqa: E402
                                          DEFAULT_VEHICLE_PARAMS, DT, VEHICLE_KINDS,
                                          LeVanLuongSimulator, road_lengths,
                                          turn_lane_bounds)

_spec = importlib.util.spec_from_file_location("gama_demand", REPO / "train" / "logic" / "demand.py")
gamademand = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gamademand)

SCRIPT = Path(__file__).resolve()
RESULTS = REPO / "validation" / "results"
HORIZON = 8000
STAT_SEEDS = list(range(1001, 1021))
HASH_SEEDS = [1001, 1002, 10_000_001, 20_000_000]
REPLAY_SEEDS = [12, 1001, 1002]
REPLAY_HORIZON = 300
SIGMA = 5.0
FLOW, MOTO, P_LEFT, P_RIGHT, HEAVY = 2400.0, 0.85, 0.2, 0.2, DEFAULT_HEAVY_RATIO
# numpy<2 has no wheel for Python 3.14 (it would be built from source); demand code does not use numpy.
CROSS_VERSIONS = {"3.10": "numpy<2", "3.11": "numpy<2", "3.14": "numpy"}

IMPLS = {
    "python": lambda path, seed, horizon: pydemand.generate_demand(path, seed, horizon),
    "gama": lambda path, seed, horizon: gamademand.generate_demand(path, seed, horizon, site="lvl"),
}


# ----------------------------------------------------------------------------- helpers
def load_rows(path):
    with Path(path).open(newline="") as stream:
        return [[float(x) for x in row] for row in csv.reader(stream) if row]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, expected, observed, z=None, ok=None, note=""):
    if ok is None:
        ok = z is not None and abs(z) <= SIGMA
    return {"name": name, "expected": expected, "observed": observed,
            "z": None if z is None else round(z, 2), "status": "PASS" if ok else "FAIL", "note": note}


def zprop(count, trials, p):
    sd = math.sqrt(trials * p * (1 - p))
    return 0.0 if sd == 0 else (count - trials * p) / sd


def spec_for(impl):
    """Expected values per implementation, from its own documented convention."""
    if impl == "python":
        return {"p_arrival": FLOW / 3600 * DT, "cycle_min": 1, "cycle_max": HORIZON,
                "kinds": {0: MOTO, 1: 1 - MOTO - HEAVY, 2: HEAVY},
                "turns": {0: P_LEFT, 1: P_RIGHT, 2: 1 - P_LEFT - P_RIGHT},
                "turn_names": {0: "trái", 1: "phải", 2: "thẳng"},
                "speeds": {code: (DEFAULT_VEHICLE_PARAMS[k]["speed_mean"], DEFAULT_VEHICLE_PARAMS[k]["speed_sd"])
                           for code, k in enumerate(VEHICLE_KINDS)},
                "kind_names": {0: "xe máy", 1: "ô tô", 2: "xe nặng"}}
    return {"p_arrival": FLOW / 3600 * 0.1, "cycle_min": 0, "cycle_max": HORIZON - 1,
            "kinds": {0: 0.85, 1: 0.15}, "turns": {0: 0.6, 1: 0.2, 2: 0.2},
            "turn_names": {0: "thẳng", 1: "trái", 2: "phải"},
            "speeds": {0: (40.0, 5.0), 1: (50.0, 5.0)}, "kind_names": {0: "xe máy", 1: "ô tô"}}


def generate_set(impl, seeds, horizon, workdir):
    out = {}
    for seed in seeds:
        path = Path(workdir) / f"{impl}_{seed}_h{horizon}.csv"
        digest = IMPLS[impl](path, seed, horizon)
        out[seed] = (path, digest, load_rows(path))
    return out


# ----------------------------------------------------------------------------- checks
def parameter_checks(impl, data, horizon, spec):
    checks = []
    seeds = len(data)
    rows = [r for _, _, rs in data.values() for r in rs]
    n = len(rows)
    trials = seeds * horizon * 4
    checks.append(check(f"số xe tổng ({seeds} seed × {horizon} tick × 4 nhánh)",
                        round(trials * spec["p_arrival"], 1), n, zprop(n, trials, spec["p_arrival"])))
    per_arm = Counter(int(r[1]) for r in rows)
    arm_z = {ARMS[a] if impl == "python" else a: round(zprop(per_arm[a], seeds * horizon, spec["p_arrival"]), 2)
             for a in range(4)}
    checks.append(check("số xe mỗi nhánh (độ lệch lớn nhất)", round(seeds * horizon * spec["p_arrival"], 1),
                        dict(per_arm), max(arm_z.values(), key=abs), note=f"z từng nhánh: {arm_z}"))
    kinds = Counter(int(r[2]) for r in rows)
    for code, p in spec["kinds"].items():
        checks.append(check(f"tỷ lệ loại xe: {spec['kind_names'][code]}", f"{p:.3f}",
                            f"{kinds[code] / n:.4f}", zprop(kinds[code], n, p)))
    extra = set(kinds) - set(spec["kinds"])
    checks.append(check("không có loại xe ngoài danh sách", "{}", sorted(extra), ok=not extra))
    turns = Counter(int(r[4]) for r in rows)
    for code, p in spec["turns"].items():
        checks.append(check(f"tỷ lệ hướng rẽ: {spec['turn_names'][code]} (mã {code})", f"{p:.3f}",
                            f"{turns[code] / n:.4f}", zprop(turns[code], n, p)))
    for code, (mean, sd) in spec["speeds"].items():
        values = [r[3] for r in rows if int(r[2]) == code]
        m = len(values)
        avg = sum(values) / m
        std = math.sqrt(sum((v - avg) ** 2 for v in values) / (m - 1))
        checks.append(check(f"tốc độ trung bình {spec['kind_names'][code]} (km/h)", mean, round(avg, 3),
                            (avg - mean) / (sd / math.sqrt(m))))
        checks.append(check(f"độ lệch chuẩn tốc độ {spec['kind_names'][code]} (km/h)", sd, round(std, 3),
                            (std / sd - 1) / (1 / math.sqrt(2 * (m - 1)))))
    low = min(r[3] for r in rows)
    checks.append(check("tốc độ nhỏ nhất", ">= 1 km/h", round(low, 3), ok=low >= 1.0))
    return checks


def structure_checks(impl, data, horizon, spec):
    checks = []
    bad_cycle = bad_dup = bad_order = bad_arm = bad_cols = 0
    for _, (_, _, rows) in data.items():
        seen = set()
        prev = None
        for r in rows:
            if len(r) != 6:
                bad_cols += 1
            cycle, arm = int(r[0]), int(r[1])
            bad_cycle += not spec["cycle_min"] <= cycle <= spec["cycle_max"]
            bad_arm += not 0 <= arm < 4
            bad_dup += (cycle, arm) in seen
            seen.add((cycle, arm))
            bad_order += prev is not None and (cycle, arm) < prev
            prev = (cycle, arm)
    rng = f"{spec['cycle_min']}..{spec['cycle_max']}"
    checks.append(check("6 cột mỗi dòng", 0, bad_cols, ok=bad_cols == 0))
    checks.append(check(f"cycle trong {rng}", 0, bad_cycle, ok=bad_cycle == 0))
    checks.append(check("nhánh trong 0..3", 0, bad_arm, ok=bad_arm == 0))
    checks.append(check("tối đa 1 xe mỗi (cycle, nhánh)", 0, bad_dup, ok=bad_dup == 0))
    checks.append(check("sắp theo (cycle, nhánh)", 0, bad_order, ok=bad_order == 0))
    return checks


def lane_checks(impl, data):
    rows = [r for _, _, rs in data.values() for r in rs]
    if impl == "gama":
        bad = sum(int(r[5]) != 0 for r in rows)
        return [check("làn luôn bằng 0 (GAMA tự chọn làn)", 0, bad, ok=bad == 0)]
    roads = road_lengths()
    bad, used = 0, defaultdict(set)
    for r in rows:
        arm, kind, turn, lane = ARMS[int(r[1])], int(r[2]), int(r[4]), int(r[5])
        lanes = roads[f"{arm}_in"]["num_lanes"]
        occupied = pydemand.OCCUPIED[VEHICLE_KINDS[kind]]
        lo, hi = ((0, lanes - occupied) if turn == 2 else
                  turn_lane_bounds(lanes, occupied, ("left", "right", "through")[turn]))
        bad += not lo <= lane <= hi
        used[(arm, kind, turn)].add(lane)
    return [check("làn nằm trong giới hạn của hướng rẽ và loại xe", 0, bad, ok=bad == 0,
                  note=f"{len(used)} tổ hợp (nhánh, loại, hướng) đã gặp")]


def seed_data_checks(impl, data, workdir):
    checks = []
    mismatch = sum(digest != sha(path) for path, digest, _ in data.values())
    checks.append(check("hash hàm trả về = sha256 của file", 0, mismatch, ok=mismatch == 0))
    if impl == "python":
        failures = []
        for path, _, _ in data.values():
            try:
                pydemand.read_demand(path)
            except ValueError as exc:
                failures.append(str(exc))
        checks.append(check("read_demand đọc lại không lỗi", 0, len(failures), ok=not failures,
                            note=failures[0] if failures else ""))
        replay = {}
        for seed in REPLAY_SEEDS:
            path = Path(workdir) / f"replay_{seed}.csv"
            pydemand.generate_demand(path, seed, REPLAY_HORIZON)
            requested = len(load_rows(path))
            sim = LeVanLuongSimulator(seed=999, demand_file=path)
            sim.step(REPLAY_HORIZON)
            replay[seed] = (requested, sim.created)
        checks.append(check(f"replay (horizon {REPLAY_HORIZON}): số xe simulator tạo = số dòng CSV",
                            {s: v[0] for s, v in replay.items()}, {s: v[1] for s, v in replay.items()},
                            ok=all(a == b for a, b in replay.values())))
    else:
        path = next(iter(data.values()))[0]
        try:
            pydemand.read_demand(path)
            note, ok = "đọc được", True
        except ValueError as exc:
            note, ok = f"bị từ chối: {exc}", False
        checks.append({"name": "file GAMA qua read_demand của Python (thông tin)", "expected": "-",
                       "observed": note, "z": None, "status": "INFO", "note": "không dùng chéo"})
    return checks


def hashes_here(horizon=HORIZON):
    """Hash for fixed seeds under the current interpreter (used in subprocesses)."""
    out = {"python_version": sys.version.split()[0], "hashes": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for impl in IMPLS:
            for seed in HASH_SEEDS:
                path = Path(tmp) / f"{impl}_{seed}.csv"
                digest = IMPLS[impl](path, seed, horizon)
                out["hashes"][f"{impl}/{seed}"] = [digest, len(load_rows(path))]
    return out


def run_hash_process(label, cmd, env_extra=None, cwd=None):
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    env.update(env_extra or {})
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"label": label, "error": str(exc)}
    if done.returncode != 0:
        return {"label": label, "error": (done.stderr.strip().splitlines() or ["lỗi không rõ"])[-1]}
    result = json.loads(done.stdout.strip().splitlines()[-1])
    result["label"] = label
    return result


def determinism(workdir):
    checks, runs = [], []
    reference = hashes_here()
    runs.append({"label": "process chính", **reference})
    again = hashes_here()
    checks.append(check("cùng seed, sinh 2 lần trong 1 process: cùng hash", "trùng",
                        "trùng" if again["hashes"] == reference["hashes"] else "khác",
                        ok=again["hashes"] == reference["hashes"]))
    for hs in ("0", "1", "12345"):
        runs.append(run_hash_process(f"subprocess PYTHONHASHSEED={hs}", [sys.executable, str(SCRIPT), "--hashes"],
                                     {"PYTHONHASHSEED": hs}))
    for version, numpy_spec in CROSS_VERSIONS.items():
        runs.append(run_hash_process(
            f"Python {version} (uv)",
            ["uv", "run", "--no-project", "--python", version, "--with", numpy_spec, "python", str(SCRIPT), "--hashes"],
            cwd=workdir))
    for run in runs[1:]:
        if "error" in run:
            checks.append({"name": f"so hash: {run['label']}", "expected": "trùng", "observed": "không chạy được",
                           "z": None, "status": "SKIP", "note": run["error"]})
        else:
            same = run["hashes"] == reference["hashes"]
            checks.append(check(f"so hash: {run['label']} (Python {run['python_version']})", "trùng",
                                "trùng" if same else "khác", ok=same))
    values = [v[0] for v in reference["hashes"].values()]
    checks.append(check("khác seed thì khác hash", f"{len(values)} hash khác nhau",
                        f"{len(set(values))} hash khác nhau", ok=len(set(values)) == len(values)))
    return checks, runs


# ----------------------------------------------------------------------------- report
def verdict(checks):
    if any(c["status"] == "FAIL" for c in checks):
        return "KHÔNG ĐẠT"
    return "ĐẠT"


def md_table(checks):
    lines = ["| Kiểm tra | Mong đợi | Đo được | z | Kết quả |", "|---|---|---|---|---|"]
    for c in checks:
        z = "" if c["z"] is None else c["z"]
        extra = f" ({c['note']})" if c["note"] else ""
        lines.append(f"| {c['name']}{extra} | {c['expected']} | {c['observed']} | {z} | {c['status']} |")
    return "\n".join(lines)


def commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "?"


def full_run():
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for impl in IMPLS:
            spec = spec_for(impl)
            data = generate_set(impl, STAT_SEEDS, HORIZON, tmp)
            results[impl] = {
                "params": parameter_checks(impl, data, HORIZON, spec),
                "structure": structure_checks(impl, data, HORIZON, spec) + lane_checks(impl, data),
                "seed_data": seed_data_checks(impl, data, tmp)}
        det, runs = determinism(tmp)
    results["determinism"] = det
    results["hash_runs"] = runs
    return results


def write_report(res):
    RESULTS.mkdir(exist_ok=True)
    py, ga = res["python"], res["gama"]
    q1 = {k: verdict(res[k]["params"] + res[k]["structure"]) for k in ("python", "gama")}
    q2 = {k: verdict(res[k]["seed_data"]) for k in ("python", "gama")}
    q3 = verdict(res["determinism"])
    reference = res["hash_runs"][0]
    skipped = [c for c in res["determinism"] if c["status"] == "SKIP"]
    out = [
        "# Báo cáo kiểm tra hàm generate demand", "",
        f"Ngày chạy: {datetime.date.today()} · commit `{commit()}` · Python {platform.python_version()} · "
        f"lệnh: `uv run --no-project --python 3.12 --with \"numpy<2\" python validation/check_demand.py`", "",
        "## 1. Kết luận", "",
        "| Câu hỏi | `python_simulation.demand` (train LVL dùng) | `train/logic/demand.py` site=lvl (GAMA) |",
        "|---|---|---|",
        f"| Thông số sinh ra đúng chưa | {q1['python']} | {q1['gama']} |",
        f"| Seed data chính xác chưa (hash, đọc lại, replay) | {q2['python']} | {q2['gama']} |",
        f"| Cùng seed ra cùng data không (cùng process, khác process, khác `PYTHONHASHSEED`, khác Python) | {q3} | {q3} |",
        "", f"Dung sai thống kê: {SIGMA:g} sigma. {len(STAT_SEEDS)} seed ({STAT_SEEDS[0]}..{STAT_SEEDS[-1]}), "
        f"horizon {HORIZON}.", "",
    ]
    if skipped:
        out += ["Không chạy được: " + "; ".join(c["name"] for c in skipped) + ". Xem mục 5.", ""]
    out += [
        "## 2. Tham số sinh", "",
        f"- Python: flow {FLOW:g} xe/h mỗi nhánh, DT {DT} s ⇒ p = {FLOW / 3600 * DT:.4f} mỗi tick mỗi nhánh; "
        f"xe máy {MOTO}, xe nặng {HEAVY}, ô tô {1 - MOTO - HEAVY:.2f}; trái {P_LEFT}, phải {P_RIGHT}; "
        "mã rẽ 0=trái 1=phải 2=thẳng; cycle bắt đầu từ 1.",
        f"- GAMA (site lvl): p = {FLOW / 3600 * 0.1:.4f} mỗi bước 0,1 s mỗi nhánh (cùng 2400 xe/h); "
        "xe máy 0.85 / ô tô 0.15; mã rẽ 0=thẳng 1=trái 2=phải (theo `collision_avoidance.gaml:481`); "
        "cycle bắt đầu từ 0; làn luôn 0.", "",
        "### Giải nghĩa các thuật ngữ", "",
        "| Từ | Nghĩa | Giá trị mặc định |", "|---|---|---|",
        "| `DT` | Bước thời gian của simulator Python: mỗi tick mô phỏng trôi 0,2 giây | 0,2 s |",
        "| tick / `cycle` | Số thứ tự bước mô phỏng, **không phải** chu kỳ đèn. Cột đầu của CSV. Tick n ứng với giây n × DT | Python từ 1; GAMA từ 0 |",
        "| `horizon` | Số tick sinh demand. 8000 tick × 0,2 s = 1600 s (khoảng 27 phút) | 8000 trong lần kiểm này |",
        "| `seed` | Số nguyên đưa vào `random.Random(seed)`. Cùng seed ⇒ cùng dãy số ngẫu nhiên ⇒ cùng file CSV | 1001..1020 (thống kê) |",
        "| nhánh / `arm` | Hướng vào ngã tư: NE, SE, SW, NW. Trong CSV là số 0..3 theo thứ tự đó | 4 nhánh |",
        "| `flow_per_approach` | Lưu lượng đến mỗi nhánh, xe/giờ | 2400 |",
        "| xác suất đến `p` | Mỗi tick, mỗi nhánh tung một lần: có xe mới với xác suất p = flow / 3600 × DT. 2400/3600 × 0,2 = 0,1333, trung bình 1 xe mỗi 7,5 tick (1,5 s) | 0,1333 (GAMA: 0,0667 vì tick 0,1 s) |",
        "| Bernoulli | Phép thử có/không với xác suất p. Số xe sau N phép thử tuân theo phân phối nhị thức Binomial(N, p) | |",
        "| `motorbike_ratio` | Tỷ lệ xe máy trong số xe sinh ra | 0,85 |",
        "| `heavy_ratio` | Tỷ lệ xe nặng (xe tải, xe buýt) | 0,03 |",
        "| ô tô | Phần còn lại: 1 − 0,85 − 0,03 | 0,12 |",
        "| `p_left`, `p_right` | Xác suất mỗi xe rẽ trái, rẽ phải; còn lại đi thẳng | 0,2 và 0,2 (thẳng 0,6) |",
        "| mã loại xe | Cột 3 của CSV: 0 xe máy, 1 ô tô, 2 xe nặng | |",
        "| mã hướng rẽ | Cột 5 của CSV. Python: 0 trái, 1 phải, 2 thẳng. GAMA: 0 thẳng, 1 trái, 2 phải | |",
        "| tốc độ | Cột 4 của CSV, km/h, bốc từ phân phối chuẩn gauss(μ, σ), tối thiểu 1 km/h | xe máy 40 ± 5, ô tô 50 ± 5, xe nặng 50 ± 2,5 |",
        "| làn / sublane | Cột 6 của CSV: chỉ số làn con tính từ lề phải, mỗi làn con rộng 1,05 m. Xe máy chiếm 1, ô tô 2, xe nặng 3 làn con. Làn bốc ngẫu nhiên trong giới hạn của hướng rẽ | |",
        "| sigma, z | Độ lệch giữa số đo và mong đợi, tính theo đơn vị độ lệch chuẩn của sai số ngẫu nhiên. |z| ≤ 5 là chấp nhận | 5 |",
        "| sha256, hash | Chuỗi 64 ký tự tóm tắt nội dung file. Chỉ cần sai một byte là hash khác hoàn toàn | |",
        "| `PYTHONHASHSEED` | Biến môi trường làm thứ tự duyệt set/dict của chuỗi thay đổi giữa các lần chạy Python. Dùng để chứng minh kết quả không phụ thuộc thứ tự đó | 0, 1, 12345 |",
        "",
        "## 3. Tiêu chí so sánh và kiểm tra", "",
        "Giá trị mong đợi lấy từ tham số trong mã của từng hàm (mục 2), không lấy từ kết quả đo. "
        "PASS = đạt tiêu chí của dòng đó; chỉ cần một dòng FAIL thì cả nhóm là KHÔNG ĐẠT.", "",
        "| Nhóm | Cách so sánh | Tiêu chí PASS |", "|---|---|---|",
        f"| Số xe tổng, số xe mỗi nhánh | Mỗi (seed, tick, nhánh) là một phép thử Bernoulli xác suất p. "
        "Số xe n ~ Binomial(N, p). z = (n − N·p) / √(N·p·(1−p)) | \\|z\\| ≤ " + f"{SIGMA:g}"
        + " (mỗi nhánh lấy nhánh lệch nhất) |",
        "| Tỷ lệ loại xe, tỷ lệ hướng rẽ | Với n xe sinh ra, số xe mỗi loại ~ Binomial(n, p loại đó). Cùng công thức z | "
        f"\\|z\\| ≤ {SIGMA:g}; không được có loại xe ngoài danh sách |",
        "| Tốc độ trung bình | z = (trung bình đo − μ) / (σ/√m), m = số xe loại đó | " + f"\\|z\\| ≤ {SIGMA:g} |",
        "| Độ lệch chuẩn tốc độ | z = (s/σ − 1) · √(2(m−1)) (xấp xỉ phân phối của độ lệch chuẩn mẫu) | "
        f"\\|z\\| ≤ {SIGMA:g} |",
        "| Tốc độ nhỏ nhất | So với sàn của hàm sinh (Python ≥ 1 km/h) | Không dòng nào dưới sàn |",
        "| Cấu trúc file | Đếm dòng vi phạm trên mọi file: 6 cột, cycle trong khoảng cho phép, nhánh 0..3, "
        "tối đa 1 xe mỗi (cycle, nhánh), sắp theo (cycle, nhánh) | 0 dòng vi phạm |",
        "| Làn | Mỗi dòng so với `turn_lane_bounds` (rẽ) hoặc 0..num_lanes − số làn xe chiếm (thẳng); bản GAMA phải bằng 0 | "
        "0 dòng ngoài giới hạn |",
        "| Hash = file | sha256 của file đọc lại so với giá trị hàm trả về | Trùng cho mọi seed |",
        "| `read_demand` | Đọc lại từng file bằng bộ đọc/kiểm của simulator Python | Không ném lỗi |",
        "| Replay | `LeVanLuongSimulator(demand_file=...)` chạy `horizon` tick, so `sim.created` với số dòng CSV | Bằng nhau |",
        "| Tất định | So `sha256` và số dòng của 4 seed cố định × 2 hàm (horizon 8000) giữa: sinh lần 2 trong cùng process, "
        "subprocess `PYTHONHASHSEED` 0 / 1 / 12345, Python 3.10 / 3.11 / 3.14 | Mọi hash trùng hoàn toàn (so từng byte qua sha256), "
        "không dùng dung sai |",
        "| Khác seed | Đếm số hash khác nhau trong 8 hash (4 seed × 2 hàm) | Mọi hash khác nhau |", "",
        f"Chọn {SIGMA:g} sigma để giảm báo động giả: có khoảng 25 phép so sánh thống kê mỗi lần chạy, "
        "ngưỡng 5 sigma cho xác suất báo nhầm cỡ 1 phần 1,7 triệu mỗi phép. Đổi lại, với cỡ mẫu "
        f"{len(STAT_SEEDS)} seed (khoảng 85 000 xe), tỷ lệ lệch dưới khoảng 0,6 đến 0,7 điểm % có thể không bị phát hiện. Kiểm chứng độ nhạy: chạy `--negative-control` "
        "(sinh với `motorbike_ratio=0.5`) phải ra FAIL.", "",
        "Mục 'Mong đợi' trong bảng kết quả: tỷ lệ hiển thị theo p; số xe hiển thị là N·p; tốc độ hiển thị là μ hoặc σ "
        "của `DEFAULT_VEHICLE_PARAMS` (Python) hoặc của mã GAMA. Cột z để trống ở các kiểm tra đếm (đạt khi bằng 0).", "",
        "## 4. Kết quả chi tiết", "",
    ]
    for key, title in (("python", "`python_simulation.demand` (Python)"), ("gama", "`train/logic/demand.py` (GAMA)")):
        out += [f"### {title}", "", "**Thông số**", "", md_table(res[key]["params"]), "",
                "**Cấu trúc file**", "", md_table(res[key]["structure"]), "",
                "**Seed data**", "", md_table(res[key]["seed_data"]), ""]
    out += ["### Tính tất định", "", md_table(res["determinism"]), "",
            "**Hash theo từng lần chạy** (sha256, số dòng; horizon 8000)", ""]
    labels = [f"{impl}/{seed}" for impl in IMPLS for seed in HASH_SEEDS]
    header = ["Lần chạy"] + labels
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run in res["hash_runs"]:
        if "error" in run:
            out.append(f"| {run['label']} | " + " | ".join(["không chạy được"] * len(labels)) + " |")
        else:
            cells = [f"`{run['hashes'][l][0][:12]}` ({run['hashes'][l][1]})" for l in labels]
            out.append(f"| {run['label']} (Python {run['python_version']}) | " + " | ".join(cells) + " |")
    out += ["", "## 5. Ghi chú và điểm chưa kiểm", "",
            "- Hai hàm mã hóa hướng rẽ khác nhau (Python 0=trái, GAMA 0=thẳng). File sinh từ hàm GAMA không đưa vào "
            "simulator Python.",
            "- `train/logic/lvl_env.py:18` trỏ tới `gama-simulation/simulation/models/LeVanLuong.gaml`, file này không "
            "có trong repo; quy ước mã rẽ của GAMA được lấy từ `train/benchmarks/gama/includes/collision_avoidance.gaml`, "
            "chưa đối chiếu với model LVL.",
            "- Ánh xạ nhánh của bản GAMA (slot 0..3 ⇒ E,N,W,S theo chú thích; model đọc N,S,E,W) chưa kiểm chứng; "
            "kiểm tra này chỉ xác nhận số xe mỗi nhánh đều nhau.",
            "- Tham số sinh (flow, tỷ lệ, p_left/p_right) không được lưu cùng seed trong `training_config.json`; "
            "report này kiểm giá trị mặc định đang có trong mã.",
            "- Không kiểm cách `train.py` gán seed cho episode / worker (seed_base + ep, validation seed); "
            "chỉ kiểm hàm sinh demand khi đã có seed.",
            "- Replay vào simulator chỉ thử với horizon 300 và 3 seed (chạy 8000 tick × nhiều seed rất chậm).",
            "- Kiểm tra thống kê chỉ phát hiện lệch lớn hơn dung sai; không chứng minh phân phối đúng từng chi tiết.",
            "- Hash chỉ chứng minh cùng kết quả trên các phiên bản Python đã chạy (xem bảng hash).", "",
            "## 6. Chạy lại", "", "```bash",
            "uv run --no-project --python 3.12 --with \"numpy<2\" python validation/check_demand.py",
            "uv run --no-project --python 3.12 --with \"numpy<2\" python validation/check_demand.py --negative-control  # phải ra FAIL", "```", ""]
    (RESULTS / "demand_check_report.md").write_text("\n".join(out), encoding="utf-8")
    (RESULTS / "demand_check_report.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    return q1, q2, q3


def negative_control():
    """Generate with motorbike_ratio=0.5 and compare with the 0.85 expectation: must FAIL."""
    spec = spec_for("python")
    with tempfile.TemporaryDirectory() as tmp:
        data = {}
        for seed in STAT_SEEDS:
            path = Path(tmp) / f"nc_{seed}.csv"
            digest = pydemand.generate_demand(path, seed, HORIZON, motorbike_ratio=0.5)
            data[seed] = (path, digest, load_rows(path))
        checks = parameter_checks("python", data, HORIZON, spec)
    failed = [c for c in checks if c["status"] == "FAIL"]
    for c in failed:
        print(f"FAIL  {c['name']}: mong đợi {c['expected']}, đo {c['observed']}, z={c['z']}")
    print("negative control:", "bị phát hiện (đúng)" if failed else "KHÔNG bị phát hiện (kiểm tra vô dụng)")
    return 0 if failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hashes", action="store_true", help="in hash cố định (dùng cho subprocess)")
    parser.add_argument("--negative-control", action="store_true", help="tham số sai phải bị phát hiện")
    args = parser.parse_args()
    if args.hashes:
        print(json.dumps(hashes_here()))
        return 0
    if args.negative_control:
        return negative_control()
    res = full_run()
    q1, q2, q3 = write_report(res)
    print("Thông số:", q1, "| Seed data:", q2, "| Tất định:", q3)
    print("Report:", RESULTS / "demand_check_report.md")
    failed = any(c["status"] == "FAIL" for k in ("python", "gama") for part in res[k].values() for c in part)
    failed = failed or any(c["status"] == "FAIL" for c in res["determinism"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
