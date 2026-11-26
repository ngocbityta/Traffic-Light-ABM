"""Learner throughput: decisions per second the Rainbow learner can absorb.

Times what LearnerThread.run does per rollout: store each transition, one
update per emitted transition (replay_ratio 1), flush the n-step tail and
publish the weights for the workers. Same network and C51 support as the
5M runs; observations are sparse like real traffic grids.

    python train/benchmarks/bench_learner.py --device cuda            # eager learner
    python train/benchmarks/bench_learner.py --device cuda --fast     # --fast-learner
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train.logic.rainbow_agent import AUX_SIZE, GRID_SHAPE, RainbowAgent  # noqa: E402
from train.logic.train import learn_emitted, weights_payload  # noqa: E402

MASK = np.array([True, False, True, False])


def observation(rng):
    grid = np.zeros(GRID_SHAPE, dtype=np.float32)
    occupied = rng.random(GRID_SHAPE) < .06
    grid[occupied] = rng.random(int(occupied.sum()))
    grid[:, 4] = np.where(occupied[:, 4], grid[:, 4] * 2 - 1, 0)
    return {"grid": grid, "aux": rng.random(AUX_SIZE).astype(np.float32)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--fill", type=int, default=6000, help="transitions stored before timing")
    p.add_argument("--rollouts", type=int, default=40, help="timed rollouts of 16 decisions")
    args = p.parse_args()
    torch.set_num_threads(1)
    agent = RainbowAgent(buffer_size=100_000, warmup=2000, atoms=201, v_min=-8000.0, v_max=0.0,
                         reward_mode="r1", device=torch.device(args.device), seed=1)
    if args.fast:
        agent.enable_fast_learner()
    run_args = SimpleNamespace(algorithm="rainbow", replay_ratio=1.0)
    rng = np.random.default_rng(0)
    pool = [observation(rng) for _ in range(64)]  # build inputs outside the timed loop

    def rollout(decisions=16, learn=True):
        for _ in range(decisions):
            emitted = agent.store(pool[rng.integers(64)], int(rng.choice([0, 2])),
                                  -float(rng.random() * 50), pool[rng.integers(64)], False,
                                  seconds=10.0, next_mask=MASK)
            if learn:
                learn_emitted(agent, run_args, emitted)
        emitted = agent.finish_episode()
        if learn:
            learn_emitted(agent, run_args, emitted)
            weights_payload(agent, 0)

    started = time.perf_counter()
    while len(agent.buffer) < args.fill:
        rollout(learn=False)
    print(f"filled {len(agent.buffer)} transitions in {time.perf_counter() - started:.1f}s")
    for _ in range(4):  # warm-up (CUDA init, graph capture)
        rollout()
    if args.device == "cuda":
        torch.cuda.synchronize()
    updates = agent.update_count
    started = time.perf_counter()
    for _ in range(args.rollouts):
        rollout()
    if args.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    done = agent.update_count - updates
    mode = "fast" if args.fast else "eager"
    graph = " (CUDA graph)" if agent.use_graph and agent._graph is not None else ""
    print(f"[{args.device} {mode}{graph}] {done} decisions in {seconds:.2f}s = "
          f"{done / seconds:.1f} decisions/s ({seconds / done * 1000:.2f} ms each)")


if __name__ == "__main__":
    main()
