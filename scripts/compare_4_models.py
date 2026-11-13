#!/usr/bin/env python3
"""
Benchmark 4 thuật toán điều khiển đèn giao thông trên GAMA:

    Fixed-Time       -> GAMA headless batch (my_batch_experiment, 8000 cycles)
    MaxPressure      -> như trên
    Q-Learning       -> như trên
    Deep Q-Learning  -> gama-server (port 6868) + train/logic/evaluate.py,
                        chính sách greedy (eps=0) với weights đã train

Quy trình:
    1. Chạy N lần độc lập cho mỗi thuật toán (--runs, mặc định 5).
    2. Tổng hợp bảng 6 tiêu chí dạng mean ± std, chọn thuật toán tốt nhất.
    3. Vẽ biểu đồ 6 panel từ series đại diện (run có throughput ở median).

Mỗi run dùng một seed ghép cặp và xác minh seed thực tế của mô phỏng.
Cùng seed chưa đảm bảo cùng demand khi controller và sinh xe dùng chung RNG.

Kết quả (thư mục comparison_results/):
    - {fixed_time,maxpressure,q_learning,dqn}_metrics.csv : series đại diện
    - traffic_comparison_4models_6metrics.png             : biểu đồ 6 panel
    - bảng mean ± std in ra console

Cách chạy:
    python3 scripts/compare_4_models.py --runs 5 --weights train/results/train_<timestamp>/best_weights.pt
    python3 scripts/compare_4_models.py --plot-only   # tổng hợp + vẽ từ CSV cũ
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------- Cấu hình ---
BASE_DIR = str(Path(__file__).resolve().parents[1])
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_results")

GAMA_HEADLESS = os.environ.get(
    "GAMA_HEADLESS", "/Applications/Gama.app/Contents/headless/gama-headless.sh"
)
BATCH_EXPERIMENT = "my_batch_experiment"
HORIZON_CYCLES = 8000        # mỗi lần chạy: 8000 cycles = 800 giây mô phỏng
FIRST_SEED = 5001
N_RUNS_DEFAULT = 5

# Tên hiển thị -> (file GAML batch, tên file CSV kết quả)
BATCH_MODELS = {
    "Fixed-Time": ("intersection-fixed-time.gaml", "fixed_time"),
    "MaxPressure": ("intersection-maxpressure.gaml", "maxpressure"),
    "Q-Learning": ("intersection-qlearning.gaml", "q_learning"),
}
DQN_NAME = "Deep Q-Learning"
MODEL_NAMES = [*BATCH_MODELS, DQN_NAME]

DQN_DIR = os.path.join(BASE_DIR, "train")
DQN_PYTHON = (os.path.join(DQN_DIR, "venv", "bin", "python")
              if os.path.isfile(os.path.join(DQN_DIR, "venv", "bin", "python"))
              else sys.executable)
DQN_EVAL = None
DQN_PORT = 6868              # gama-server chạy sẵn (kiểm tra: lsof -i :6868)

# Dòng "METRIC_CSV,cycle,passed,avg_wait,queue,speed,travel,max_wait" mà
# các model GAML in ra mỗi 10 cycles
METRIC_CSV_RE = re.compile(
    r"METRIC_CSV,(\d+),(\d+),([\d.]+),(\d+),([\d.]+),([\d.]+),([\d.]+)")


def validate_sampling_grid(df, label):
    expected = list(range(10, HORIZON_CYCLES, 10))
    actual = df["cycle"].astype(int).tolist()
    if actual != expected:
        raise ValueError(
            f"[{label}] sampling grid differs from cycles 10..{HORIZON_CYCLES - 10} "
            f"at intervals of 10 (got {len(actual)} rows, first={actual[:3]}, "
            f"last={actual[-3:]})"
        )

CHART_PATH = os.path.join(OUTPUT_DIR, "traffic_comparison_4models_6metrics.png")
COLORS = {"Fixed-Time": "#e74c3c", "MaxPressure": "#2ecc71",
          "Q-Learning": "#3498db", DQN_NAME: "#9b59b6"}


# --------------------------------------------------- Chạy 1 model batch GAMA
def run_batch(model_file, label, seed):
    """Chạy 1 lần batch headless, trả về DataFrame metrics (799 dòng)."""
    print(f"🚀 [{label}] batch headless, {HORIZON_CYCLES} cycles...", flush=True)
    source = f'''model seeded_benchmark
import "{Path(model_file).name}"
global {{
    reflex benchmark_seed_log when: cycle = 0 {{ write "BENCHMARK_SEED," + seed; }}
}}
experiment seeded_benchmark_experiment type: batch until: (cycle >= {HORIZON_CYCLES}) {{
    parameter "Traffic seed" var: traffic_seed init: {float(seed)};
}}
'''
    with tempfile.NamedTemporaryFile("w", suffix=".gaml", prefix="tmp_seeded_benchmark_",
                                     dir=Path(model_file).parent, delete=False) as stream:
        stream.write(source)
        wrapper = Path(stream.name)
    cmd = ["bash", GAMA_HEADLESS, "-m", "2048m", "-batch",
           "seeded_benchmark_experiment", str(wrapper)]
    records = []
    actual_seed = None
    log_path = Path(OUTPUT_DIR) / f"{Path(model_file).stem}_seed{seed}.log"
    try:
        with tempfile.TemporaryDirectory(prefix="gama_benchmark_") as work, log_path.open("w") as log:
            with subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  env={**os.environ, "_JAVA_OPTIONS": os.environ.get("_JAVA_OPTIONS", "") +
                                       " -Xms256m -Xmx2g -XX:-AlwaysPreTouch"}) as process:
                for line in process.stdout:
                    log.write(line)
                    if line.startswith("BENCHMARK_SEED,"):
                        actual_seed = float(line.split(",", 1)[1])
                    m = METRIC_CSV_RE.search(line)
                    if not m:
                        continue
                    cycle = int(m.group(1))
                    records.append({
                        "seed": seed, "cycle": cycle,
                        "time_sec": cycle * 0.1,
                        "throughput": int(m.group(2)),
                        "avg_waiting_time": float(m.group(3)),
                        "queue_length": int(m.group(4)),
                        "avg_speed": float(m.group(5)),
                        "avg_travel_time": float(m.group(6)),
                        "max_waiting_time": float(m.group(7)),
                    })
            if process.returncode != 0:
                raise RuntimeError(f"[{label}] GAMA exited with code {process.returncode}; see {log_path}")
    finally:
        wrapper.unlink(missing_ok=True)
    if actual_seed != seed:
        raise RuntimeError(f"[{label}] requested seed {seed}, actual seed {actual_seed}; see {log_path}")

    if not records:
        raise RuntimeError(f"[{label}] Không thu được dòng METRIC_CSV nào - "
                           f"kiểm tra model GAML hoặc tên experiment.")
    df = pd.DataFrame(records)
    validate_sampling_grid(df, label)
    print(f"✅ [{label}] {len(df)} mẫu (cycle {df['cycle'].min()}"
          f"-{df['cycle'].max()}), throughput={df['throughput'].iloc[-1]}",
          flush=True)
    return df


# ------------------------------------------------------- Đánh giá Deep Q-Net
def run_dqn(n_episodes, weights_path):
    """Chạy evaluate.py qua gama-server, trả về list DataFrame mỗi episode."""
    print(f"🚀 [{DQN_NAME}] greedy (eps=0) qua gama-server :{DQN_PORT}, "
          f"{n_episodes} episodes (seed {FIRST_SEED}..{FIRST_SEED + n_episodes - 1})...",
          flush=True)
    cmd = [DQN_PYTHON, "-m", "train.logic.evaluate",
           "--scenario", "legacy", "--backend", "gama",
           "--weights", weights_path,
           "--episodes", str(n_episodes),
           "--episode-horizon", str(HORIZON_CYCLES),
           "--seed-base", str(FIRST_SEED - 1),   # ep i -> seed = FIRST_SEED + i - 1
           "--port", str(DQN_PORT),
           "--host", "localhost",
           "--output", os.path.join(OUTPUT_DIR, "dqn_metrics.csv")]
    process = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(process.stdout.readline, ""):
        print(f"   [DQN] {line.rstrip()}", flush=True)
    process.stdout.close()
    process.wait(timeout=60)

    if process.returncode != 0:
        raise RuntimeError(f"[{DQN_NAME}] evaluate.py thoát với mã "
                           f"{process.returncode} - gama-server đang chạy ở "
                           f"port {DQN_PORT} không?")

    csv_path = os.path.join(OUTPUT_DIR, "dqn_metrics.csv")
    df = pd.read_csv(csv_path)
    episodes = [g.reset_index(drop=True) for _, g in df.groupby("episode")]
    if len(episodes) != n_episodes:
        raise RuntimeError(f"[{DQN_NAME}] expected {n_episodes} episodes, got {len(episodes)}")
    for i, episode in enumerate(episodes, 1):
        validate_sampling_grid(episode, f"{DQN_NAME} episode {i}")
    print(f"✅ [{DQN_NAME}] {len(episodes)} episode -> {csv_path}", flush=True)
    return episodes


# ------------------------------------------------------------- Tổng hợp ------
# (key cột, nhãn bảng, giá trị cao hơn là tốt hơn)
METRICS = [
    ("throughput", "1. Throughput (Tổng xe qua)", True),
    ("avg_waiting_time", "2. Average Waiting Time", False),
    ("queue_length", "3. Average Queue Length", False),
    ("avg_speed", "4. Average Speed", True),
    ("avg_travel_time", "5. Average Travel Time", False),
    ("max_waiting_time", "6. Max Waiting Time (Fairness)", False),
]


def per_run_values(df):
    """Giá trị 6 tiêu chí của MỘT lần chạy:
    throughput & travel time lấy giá trị cuối, max wait lấy max,
    wait/queue/speed lấy trung bình theo thời gian."""
    return {
        "throughput": df["throughput"].iloc[-1],
        "avg_waiting_time": df["avg_waiting_time"].mean(),
        "queue_length": df["queue_length"].mean(),
        "avg_speed": df["avg_speed"].mean(),
        "avg_travel_time": df["avg_travel_time"].iloc[-1],
        "max_waiting_time": df["max_waiting_time"].max(),
    }


def mean_std(values):
    m = sum(values) / len(values)
    sd = (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5
    return m, sd


def report(reps, n_runs):
    """In bảng 6 tiêu chí dạng mean ± std trên n_runs lần chạy, chọn tốt nhất."""
    runs = {name: [per_run_values(df) for df in reps[name]] for name in MODEL_NAMES}

    print("\n" + "=" * 140)
    print(f"  BẢNG SO SÁNH 4 THUẬT TOÁN — MEAN ± STD TRÊN {n_runs} LẦN CHẠY ĐỘC LẬP "
          f"(mỗi lần {HORIZON_CYCLES} cycles)")
    print("=" * 140)
    header = f"{'Tiêu chí':<34} |" + "".join(f"{n:>17} |" for n in MODEL_NAMES) \
             + " Tốt nhất (mean)"
    print(header)
    print("-" * 140)

    wins = {name: 0 for name in MODEL_NAMES}
    for key, label, higher_better in METRICS:
        means = {name: mean_std([r[key] for r in runs[name]])[0] for name in MODEL_NAMES}
        best = max(means, key=means.get) if higher_better else min(means, key=means.get)
        wins[best] += 1

        cells = ""
        for name in MODEL_NAMES:
            m, sd = mean_std([r[key] for r in runs[name]])
            cells += f"{m:9.2f}±{sd:<5.2f} |"
        print(f"{label:<34} |{cells} 🏆 {best}")

    print("=" * 140)
    print("  Số tiêu chí thắng (theo mean): "
          + " | ".join(f"{n}: {w}" for n, w in wins.items()))
    print("  Lưu ý: throughput/speed cao hơn là tốt; các tiêu chí còn lại thấp hơn là tốt.")


# ------------------------------------------------------------- Biểu đồ ------
def representative(df_list):
    """Chọn series đại diện để vẽ: run có throughput cuối ở vị trí median."""
    if len(df_list) == 1:
        return df_list[0]
    finals = [d["throughput"].iloc[-1] for d in df_list]
    return df_list[finals.index(sorted(finals)[len(finals) // 2])]


def save_series(df, csv_stem):
    """Lưu series đại diện của 1 model ra CSV (dùng lại cho --plot-only)."""
    path = os.path.join(OUTPUT_DIR, f"{csv_stem}_metrics.csv")
    df.to_csv(path, index=False)


def plot(series_by_model):
    """Vẽ 6 panel (3x2), mỗi panel một tiêu chí so sánh 4 thuật toán."""
    plt.style.use("seaborn-v0_8-whitegrid"
                  if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("So Sánh 4 Thuật Toán Điều Khiển Đèn Giao Thông "
                 "(Fixed-Time, MaxPressure, Q-Learning, Deep Q-Learning)\n"
                 f"8000 cycles (800s) mỗi lần chạy — GAMA Platform",
                 fontsize=15, fontweight="bold", y=0.99)

    panels = [
        ("throughput", "1. Throughput [↑ Tốt]", "Số xe"),
        ("avg_waiting_time", "2. Average Waiting Time [↓ Tốt]", "Giây (s)"),
        ("queue_length", "3. Average Queue Length [↓ Tốt]", "Số xe"),
        ("avg_speed", "4. Average Speed [↑ Tốt]", "km/h"),
        ("avg_travel_time", "5. Average Travel Time [↓ Tốt]", "Giây (s)"),
        ("max_waiting_time", "6. Max Waiting Time (Fairness) [↓ Tốt]", "Giây (s)"),
    ]
    for idx, (column, title, ylabel) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        for name in MODEL_NAMES:
            df = series_by_model[name]
            is_dqn = name == DQN_NAME
            ax.plot(df["time_sec"], df[column], label=name, color=COLORS[name],
                    lw=2, linestyle="--" if is_dqn else "-")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Thời gian mô phỏng (giây)")
        ax.set_ylabel(ylabel)
        if idx == 0:
            ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(CHART_PATH, dpi=300)
    plt.close(fig)
    print(f"\n📊 Biểu đồ so sánh 4 mô hình (300 DPI): {CHART_PATH}", flush=True)


# ------------------------------------------------------------------ main ----
def load_saved_series():
    """Mode --plot-only: đọc lại CSV đã lưu, mỗi model 1 series đại diện.
    CSV của DQN chứa nhiều episode -> chọn episode đại diện (median)."""
    series = {}
    for name in MODEL_NAMES:
        csv_stem = BATCH_MODELS.get(name, (None, "dqn"))[1]
        df = pd.read_csv(os.path.join(OUTPUT_DIR, f"{csv_stem}_metrics.csv"))
        if "episode" in df.columns and df["episode"].nunique() > 1:
            episodes = [g.reset_index(drop=True) for _, g in df.groupby("episode")]
            df = representative(episodes)
        validate_sampling_grid(df, name)
        series[name] = df.drop(columns=["episode"], errors="ignore")
    return series


def main(argv=None):
    global OUTPUT_DIR, CHART_PATH, DQN_PORT, FIRST_SEED
    parser = argparse.ArgumentParser(
        description="Benchmark 4 thuật toán điều khiển đèn giao thông trên GAMA")
    parser.add_argument("--runs", type=int, default=N_RUNS_DEFAULT,
                        help=f"Số lần chạy độc lập mỗi thuật toán "
                             f"(mặc định {N_RUNS_DEFAULT})")
    parser.add_argument("--weights",
                        help="DQN checkpoint trained with the current traffic model (required unless --plot-only)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory for benchmark CSV files and chart")
    parser.add_argument("--port", type=int, default=DQN_PORT,
                        help="Port of the running GAMA server for DQN evaluation")
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--plot-only", action="store_true",
                        help="Không chạy mô phỏng, chỉ tổng hợp + vẽ từ CSV đã lưu")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not args.plot_only and not args.weights:
        parser.error("--weights is required; train DQN on the current DoAn model first")
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    CHART_PATH = os.path.join(OUTPUT_DIR, "traffic_comparison_4models_6metrics.png")
    DQN_PORT = args.port
    FIRST_SEED = args.first_seed
    if args.weights:
        args.weights = str(Path(args.weights).resolve())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_runs = 1 if args.plot_only else args.runs

    print("=" * 140)
    print(" BENCHMARK 4 THUẬT TOÁN: FIXED-TIME | MAXPRESSURE | Q-LEARNING | "
          "DEEP Q-LEARNING")
    print(f" {n_runs} lần chạy độc lập mỗi thuật toán | "
          f"{HORIZON_CYCLES} cycles/lần | output: {OUTPUT_DIR}")
    print("=" * 140, flush=True)

    if args.plot_only:
        series = load_saved_series()
        reps = {name: [series[name]] for name in MODEL_NAMES}
    else:
        # Old checkpoints were trained before changes to vehicle dynamics or
        # observations. Comparing them with newly simulated controllers would
        # produce a misleading benchmark.
        model_sources = [
            os.path.join(BASE_DIR, "train/benchmarks/gama/includes/collision_avoidance.gaml"),
            os.path.join(BASE_DIR, "train/benchmarks/gama/includes/doan_scenery.gaml"),
            os.path.join(BASE_DIR, "train/benchmarks/gama/models/intersection-dqn-python.gaml"),
            os.path.join(DQN_DIR, "gama_env.py"),
            os.path.join(DQN_DIR, "train.py"),
        ]
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f"DQN checkpoint not found: {args.weights}")
        if os.path.getmtime(args.weights) < max(map(os.path.getmtime, model_sources)):
            raise RuntimeError(
                "DQN checkpoint predates the current simulation/training code. "
                "Retrain DQN and pass the new checkpoint with --weights."
            )
        reps = {name: [] for name in MODEL_NAMES}
        for name, (model_file, csv_stem) in BATCH_MODELS.items():
            model_path = os.path.join(BASE_DIR, "train/benchmarks/gama/models", model_file)
            reps[name] = []
            for i in range(n_runs):
                reps[name].append(run_batch(model_path, name, FIRST_SEED + i))
                pd.concat([df.assign(episode=j + 1) for j, df in enumerate(reps[name])],
                          ignore_index=True).to_csv(
                    os.path.join(OUTPUT_DIR, f"{csv_stem}_all_runs.csv"), index=False)
            save_series(representative(reps[name]), csv_stem)
        reps[DQN_NAME] = run_dqn(n_runs, args.weights)

    summaries = []
    for name, runs in reps.items():
        for i, df in enumerate(runs):
            summaries.append(dict(method=name, episode=i + 1,
                                  **per_run_values(df)))
    pd.DataFrame(summaries).to_csv(os.path.join(OUTPUT_DIR, "per_run_summary.csv"), index=False)
    if not args.plot_only:
        Path(OUTPUT_DIR, "benchmark_manifest.json").write_text(json.dumps(dict(
            seeds=list(range(FIRST_SEED, FIRST_SEED + n_runs)),
            horizon=HORIZON_CYCLES, weights=args.weights,
            limitations=["Shared RNG and rejected spawn requests can change demand between methods",
                         "Native Q-Learning trains online from zero during each evaluation run",
                         "DQN checkpoint quality must be assessed separately"]), indent=2) + "\n")

    report(reps, n_runs)
    plot({name: representative(reps[name]) for name in MODEL_NAMES})
    print(f"\n🎉 Hoàn thành benchmark ({n_runs} lần chạy/thuật toán)!")


if __name__ == "__main__":
    main()
