#!/usr/bin/env python3
"""Serve the trained benchmark DQN to GAMA Studio's GUI demo experiment.

Run ``python3 scripts/demo_dqn.py``, then open
``train/benchmarks/gama/models/intersection-dqn-demo.gaml`` and run ``my_experiment``.
The two processes exchange one request and one action per decision through
``train/results/demo_bridge``. No GAMA headless server is needed.
"""

import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BRIDGE = ROOT / "train" / "results" / "demo_bridge"
DEFAULT_WEIGHTS = ROOT / "comparison_results" / "fair_benchmark_20261005_final" / "best_dqn.pt"


def use_project_python():
    """Make the documented python3 command work when torch is only in venv."""
    try:
        import torch  # noqa: F401
    except ImportError:
        project_python = ROOT / "train" / "venv" / "bin" / "python"
        if not project_python.is_file() or Path(sys.executable).resolve() == project_python.resolve():
            raise SystemExit("Thiếu PyTorch. Chạy bash train/scripts/setup_env.sh trước.")
        os.execv(str(project_python), [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def load_network(weights):
    import torch

    sys.path.insert(0, str(ROOT / "train" / "logic"))
    from train.logic.dqn_agent import QNetwork

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if "online" not in checkpoint:
        raise ValueError("Checkpoint này không thuộc benchmark DQN 11 đầu vào.")
    network = QNetwork(11, (64, 64))
    network.load_state_dict(checkpoint["online"])
    network.eval()
    return network


def parse_request(raw):
    """Match scripts/fair_benchmark.py:features, including feature order."""
    parts = [int(value.strip()) for value in raw.strip().split(",")]
    if len(parts) != 11:
        raise ValueError(f"Cần 11 số trong request, nhận {len(parts)}")
    cycle, phase, green_cycles, *counts = parts
    if phase not in (0, 4) or cycle < 0 or not 0 <= green_cycles <= 600:
        raise ValueError(f"Điểm quyết định không hợp lệ: cycle={cycle}, phase={phase}")
    if any(value < 0 for value in counts):
        raise ValueError("Số xe không được âm")
    features = [min(value / 20.0, 3.0) for value in counts]
    features += [green_cycles / 600.0, float(phase == 0), float(phase == 4)]
    return cycle, features


def choose_action(network, features):
    import torch

    with torch.no_grad():
        scores = network(torch.tensor(features, dtype=torch.float32))
    return int(scores.argmax().item()), [round(float(value), 3) for value in scores]


def serve(network, poll_seconds):
    BRIDGE.mkdir(parents=True, exist_ok=True)
    for pattern in ("request_*.txt", "action_*.txt"):
        for path in BRIDGE.glob(pattern):
            path.unlink()
    ready = BRIDGE / "ready.txt"
    ready.write_text(f"{os.getpid()}\n")
    incomplete_since = {}
    print("DQN đã sẵn sàng.", flush=True)
    try:
        while True:
            for request in sorted(BRIDGE.glob("request_*.txt")):
                try:
                    cycle, features = parse_request(request.read_text())
                    action, scores = choose_action(network, features)
                    serial = request.stem.removeprefix("request_")
                    response = BRIDGE / f"action_{serial}.txt"
                    temporary = BRIDGE / f".action_{serial}.{os.getpid()}.tmp"
                    temporary.write_text(f"{action}\n")
                    os.replace(temporary, response)
                    request.unlink()
                except (OSError, ValueError) as error:
                    # GAMA may still be writing the request; retry next poll.
                    first_seen = incomplete_since.setdefault(request.name, time.monotonic())
                    if time.monotonic() - first_seen > 2:
                        print(f"Request chưa hoàn chỉnh ({request.name}): {error}", flush=True)
                        incomplete_since[request.name] = time.monotonic()
                    continue
                incomplete_since.pop(request.name, None)
                label = "GIỮ XANH" if action == 0 else "CHUYỂN PHA"
                print(f"cycle={cycle:5d} | Q={scores} | {label}", flush=True)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nĐã dừng DQN demo.")
    finally:
        ready.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Checkpoint best_dqn.pt của fair_benchmark.py")
    parser.add_argument("--poll-ms", type=int, default=20)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Không tìm thấy checkpoint: {args.weights}")
    if args.poll_ms < 1:
        parser.error("--poll-ms phải lớn hơn 0")
    use_project_python()
    network = load_network(args.weights)
    print(f"Đã tải chính sách đã huấn luyện: {args.weights}", flush=True)
    serve(network, args.poll_ms / 1000.0)


if __name__ == "__main__":
    main()
