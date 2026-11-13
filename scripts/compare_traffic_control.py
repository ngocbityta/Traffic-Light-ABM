#!/usr/bin/env python3
"""Compare mobility and safety metrics for the three GAMA controllers."""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GAMA_HEADLESS = Path(os.environ.get(
    "GAMA_HEADLESS", "/Applications/Gama.app/Contents/headless/gama-headless.sh"
))
MODELS = {
    "Fixed-Time": ("intersection-fixed-time.gaml", "fixed_time", "#e74c3c"),
    "MaxPressure": ("intersection-maxpressure.gaml", "maxpressure", "#2ecc71"),
    "Q-Learning": ("intersection-qlearning.gaml", "q_learning", "#3498db"),
}
MOBILITY_FIELDS = (
    "cycle", "throughput", "avg_waiting_time", "queue_length", "avg_speed",
    "avg_travel_time", "max_waiting_time",
)
SAFETY_FIELDS = (
    "cycle", "TIT_total", "DRAC_max", "TTC_lt1_count", "hard_brakes",
    "PET_lt1_5_count", "min_PET", "DZ_count",
)
MOBILITY_PATTERN = re.compile(r"METRIC_CSV,(\d+),(\d+),([\d.]+),(\d+),([\d.]+),([\d.]+),([\d.]+)")
SAFETY_PATTERN = re.compile(r"SAFETY_CSV,(\d+),([\d.]+),([\d.]+),(\d+),(\d+),(\d+),([\d.]+),(\d+)")
INTEGER_FIELDS = {
    "cycle", "throughput", "queue_length", "TTC_lt1_count", "hard_brakes",
    "PET_lt1_5_count", "DZ_count",
}

# (column, label, unit, aggregation, higher is better)
MOBILITY_METRICS = (
    ("throughput", "Throughput (Tổng xe qua)", "xe", "last", True),
    ("avg_waiting_time", "Average Waiting Time", "s", "mean", False),
    ("queue_length", "Average Queue Length", "xe", "mean", False),
    ("avg_speed", "Average Speed", "km/h", "mean", True),
    ("avg_travel_time", "Average Travel Time", "s", "last", False),
    ("max_waiting_time", "Max Waiting Time (Fairness)", "s", "max", False),
)
SAFETY_METRICS = (
    ("TIT_total", "Total TIT (Tích phân rủi ro)", "s", "last", False),
    ("TTC_lt1_count", "Severe Conflicts (TTC < 1.0s)", "lần", "last", False),
    ("hard_brakes", "Hard Braking (a < -3.4m/s²)", "lần", "last", False),
    ("PET_lt1_5_count", "PET Near-Miss (PET < 1.5s)", "lần", "last", False),
    ("min_PET", "Min Observed PET", "s", "min", True),
    ("DRAC_max", "Max Observed DRAC", "m/s²", "max", False),
    ("DZ_count", "Dilemma Zone Vehicles", "xe", "last", False),
)


def parse_record(line, pattern, fields):
    """Extract one metric record from a GAMA log line."""
    match = pattern.search(line)
    if not match:
        return None
    values = match.groups()
    cycle = int(values[0])
    record = {"cycle": cycle, "time_sec": cycle * 0.1}
    record.update(
        (field, int(value) if field in INTEGER_FIELDS else float(value))
        for field, value in zip(fields[1:], values[1:])
    )
    return record


def run_simulation(model_path, model_name, output_dir):
    """Run GAMA and save CSVs only after a successful simulation."""
    print(f"\n🚀 Đang chạy mô phỏng: {model_name}...", flush=True)
    command = ["bash", str(GAMA_HEADLESS), "-m", "2048m", "-batch", "my_batch_experiment", str(model_path)]
    mobility, safety = [], []
    batch_env = {
        **os.environ,
        "_JAVA_OPTIONS": os.environ.get("_JAVA_OPTIONS", "") + " -Xms256m -Xmx2g -XX:-AlwaysPreTouch",
    }
    with tempfile.TemporaryDirectory(prefix="gama_comparison_") as work:
        with subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1,
                              env=batch_env) as process:
            for line in process.stdout:
                metric = parse_record(line, MOBILITY_PATTERN, MOBILITY_FIELDS)
                if metric:
                    mobility.append(metric)
                    if metric["cycle"] % 400 == 0:
                        print(f"   [{model_name}] cycle {metric['cycle']}: "
                              f"passed={metric['throughput']:.0f}, queue={metric['queue_length']:.0f}")
                safety_record = parse_record(line, SAFETY_PATTERN, SAFETY_FIELDS)
                if safety_record:
                    safety.append(safety_record)
    if process.returncode != 0:
        raise RuntimeError(f"{model_name}: GAMA exited with code {process.returncode}")
    if not mobility:
        raise RuntimeError(f"{model_name}: no METRIC_CSV rows; check model and experiment")

    expected = list(range(10, 8000, 10))
    targets = [("mobility", mobility)]
    if safety:
        targets.append(("safety", safety))
    for kind, records in targets:
        if [row["cycle"] for row in records] != expected:
            raise RuntimeError(f"{model_name}: incomplete {kind} sampling grid (expected cycles 10..7990)")
    metric_df = pd.DataFrame(mobility)
    safety_df = pd.DataFrame(safety) if safety else None
    stem = MODELS[model_name][1]
    metric_df.to_csv(output_dir / f"{stem}_metrics.csv", index=False)
    if safety_df is not None:
        safety_df.to_csv(output_dir / f"{stem}_safety.csv", index=False)
    print(f"✅ {model_name}: {len(metric_df)} mobility" + (f", {len(safety_df)} safety samples" if safety_df is not None else ""))
    return metric_df, safety_df


def aggregate(df, column, method):
    series = df[column]
    if method == "last":
        return series.iloc[-1]
    return getattr(series, method)()


def print_summary(title, datasets, definitions):
    print(f"\n{title}")
    print(f"{'Metric':<38} | " + " | ".join(f"{name:>14}" for name in MODELS) + " | Best")
    for column, label, unit, method, higher_is_better in definitions:
        values = {name: aggregate(df, column, method) for name, df in datasets.items()}
        best = (max if higher_is_better else min)(values, key=values.get)
        cells = " | ".join(f"{values[name]:>10.2f} {unit:<3}" for name in MODELS)
        print(f"{label:<38} | {cells} | {best}")


def plot_mobility(datasets, path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("So Sánh Hiệu Năng Lưu Thông", fontsize=16, fontweight="bold")
    for ax, (column, label, _, _, _) in zip(axes.flat, MOBILITY_METRICS):
        for name, df in datasets.items():
            ax.plot(df["time_sec"], df[column], label=name, color=MODELS[name][2],
                    lw=2.2, linestyle="--" if name == "Q-Learning" else "-")
        ax.set(title=label, xlabel="Thời gian mô phỏng (giây)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"📊 {path}")


def plot_safety(datasets, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("So Sánh Chỉ Số An Toàn Vi Mô (SSM)", fontsize=15, fontweight="bold")
    for ax, column, title in zip(
        (axes[0, 0], axes[0, 1], axes[1, 0]),
        ("TIT_total", "TTC_lt1_count", "hard_brakes"),
        ("Total TIT [↓]", "TTC < 1.0s [↓]", "Hard Braking [↓]"),
    ):
        for name, df in datasets.items():
            ax.plot(df["time_sec"], df[column], label=name, color=MODELS[name][2], lw=2)
        ax.set(title=title, xlabel="Thời gian (giây)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    categories = ("Severe TTC", "Hard Brakes", "PET < 1.5s")
    columns = ("TTC_lt1_count", "hard_brakes", "PET_lt1_5_count")
    positions = np.arange(len(categories))
    for index, (name, df) in enumerate(datasets.items()):
        values = [df[column].iloc[-1] for column in columns]
        axes[1, 1].bar(positions + (index - 1) * 0.25, values, 0.25,
                       label=name, color=MODELS[name][2])
    axes[1, 1].set_xticks(positions, categories)
    axes[1, 1].set_title("Tổng kết chỉ số sự cố an toàn [↓]")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"📊 {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "comparison_results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mobility, safety = {}, {}
    for name, (filename, _, _) in MODELS.items():
        mobility[name], safety[name] = run_simulation(
            ROOT / "train" / "benchmarks" / "gama" / "models" / filename, name, args.output_dir)

    print_summary("6 CHỈ SỐ LƯU THÔNG", mobility, MOBILITY_METRICS)
    plot_mobility(mobility, args.output_dir / "traffic_comparison_3models_mobility.png")
    if all(df is not None and not df.empty for df in safety.values()):
        print_summary("7 CHỈ SỐ AN TOÀN VI MÔ", safety, SAFETY_METRICS)
        plot_safety(safety, args.output_dir / "traffic_comparison_3models_safety.png")


if __name__ == "__main__":
    main()
