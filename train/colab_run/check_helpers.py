"""Checks run by check_all.command on the machine that will train.

    python train/colab_run/check_helpers.py native
    python train/colab_run/check_helpers.py simhash OUT.json SEED SCALE IMB TICKS
    python train/colab_run/check_helpers.py config TRAIN_ARGS...
    python train/colab_run/check_helpers.py demand
    python train/colab_run/check_helpers.py run RUN_DIR
"""
import csv
import dataclasses
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python-simulation" / "simulation"))


def fail(message):
    print("FAIL:", message)
    sys.exit(1)


def native():
    import python_simulation.python_sim as sim
    import python_simulation.junction_negotiation as negotiation
    import python_simulation.vehicle_geometry as geometry
    for module in (sim, negotiation, geometry):
        print(f"  {module.__name__}: {Path(module.__file__).name}")
        if not module.__file__.endswith((".so", ".pyd")):
            fail(f"{module.__name__} is pure Python: build_native.py did not take effect")
    print("OK native extensions loaded")


def simhash(out, seed, scale, imbalance, ticks):
    """Per-tick hash of the whole simulator state (compare native vs pure Python)."""
    from python_simulation.python_sim import LeVanLuongSimulator, Vehicle
    from python_simulation.demand import axis_flows, generate_demand
    from python_simulation.sim_config import load_sim_config
    fields = [f.name for f in dataclasses.fields(Vehicle)]

    def enc(x):
        if isinstance(x, float):
            return x.hex()
        if isinstance(x, (tuple, list)):
            return "(" + ",".join(enc(y) for y in x) + ")"
        if x is None or isinstance(x, (int, str, bool)):
            return repr(x)
        return type(x).__name__

    cfg = load_sim_config()
    seed, ticks = int(seed), int(ticks)
    path = Path(tempfile.gettempdir()) / f"check_demand_{seed}_{scale}_{imbalance}.csv"
    generate_demand(path, seed, 8000, heavy_ratio=cfg["heavy_ratio"],
                    flow_by_arm=axis_flows(float(scale), float(imbalance)))
    sim = LeVanLuongSimulator(seed=seed, demand_file=str(path), train_mode=True, mode="safe", **cfg)
    rng = random.Random(seed)
    hashes = []
    for _ in range(ticks):
        if sim.dqn_waiting_action:
            mask = sim.action_mask()
            sim.set_action(rng.choice([i for i, ok in enumerate(mask) if ok]))
        sim.step(1)
        h = hashlib.sha256()
        for v in sim.vehicles:
            h.update(("|".join(enc(getattr(v, n)) for n in fields) + ";").encode())
        h.update(enc((sim.cycle, sim.phase, sim.created, sim.passed, sim.system_time_integral,
                      sim.queue_time_integral, sim.jam_removed)).encode())
        hashes.append(h.hexdigest()[:16])
    Path(out).write_text(json.dumps({"hashes": hashes, "passed": sim.passed,
                                     "vehicles": len(sim.vehicles)}))
    print(f"  seed {seed} scale {scale}: {ticks} ticks, passed {sim.passed}, in network {len(sim.vehicles)}")


def compare(a, b):
    ha, hb = json.loads(Path(a).read_text())["hashes"], json.loads(Path(b).read_text())["hashes"]
    for tick, (x, y) in enumerate(zip(ha, hb), 1):
        if x != y:
            fail(f"native and pure Python differ from tick {tick}")
    if len(ha) != len(hb):
        fail("different lengths")
    print(f"OK identical for {len(ha)} ticks")


def config(*train_args):
    from train.logic.train import parse_args, training_config
    args = parse_args(list(train_args))
    c = training_config(args)
    c_view = dict(c, total_steps=args.total_steps)
    keys = ("algorithm", "reward_mode", "total_steps", "episode_horizon", "gamma", "lr",
            "buffer_size", "warmup", "batch_size", "n_step", "c51_atoms", "c51_v_min",
            "c51_v_max", "demand_scale_min", "demand_scale_max", "axis_imbalance_max",
            "validation_levels", "queue_feature_scale")
    for key in keys:
        print(f"  {key} = {c_view.get(key)}")
    expected = {"algorithm": "rainbow", "reward_mode": "r1", "total_steps": 5_000_000,
                "c51_atoms": 201, "c51_v_min": -8000.0, "demand_scale_min": 0.5,
                "demand_scale_max": 1.3, "axis_imbalance_max": 0.2,
                "queue_feature_scale": "log",
                "validation_levels": [[0.5, 0.1], [0.9, 0.1], [1.3, 0.1]]}
    for key, value in expected.items():
        got = c_view.get(key)
        if isinstance(value, list):
            ok = len(got) == len(value) and all(
                abs(a - x) < 1e-9 and abs(b - y) < 1e-9 for (a, b), (x, y) in zip(got, value))
        else:
            ok = got == value
        if not ok:
            fail(f"{key} = {got}, expected {value}")
    from train.logic.train import check_resume_config
    if check_resume_config(json.loads(json.dumps(c)), c):
        fail("config does not survive the JSON round trip")
    digest = hashlib.sha256(json.dumps(c["code_sha256"], sort_keys=True).encode()).hexdigest()
    print(f"  code files hashed: {len(c['code_sha256'])}, combined {digest[:16]}")
    Path(tempfile.gettempdir(), "check_code_hash.txt").write_text(digest)
    print("OK training settings")


def demand():
    from python_simulation.demand import ARMS, axis_flows, generate_demand, sample_demand_level
    levels = [sample_demand_level(seed, 0.5, 1.3, 0.2) for seed in range(1001, 11001)]
    scales = [s for s, _ in levels]
    imbalances = [i for _, i in levels]
    print(f"  10000 episode seeds: scale {min(scales):.3f}..{max(scales):.3f} mean "
          f"{sum(scales) / len(scales):.3f}; imbalance {min(imbalances):.3f}..{max(imbalances):.3f}")
    if not (0.5 <= min(scales) < 0.51 and 1.29 < max(scales) <= 1.3 and
            abs(sum(scales) / len(scales) - 0.9) < 0.01 and max(imbalances) <= 0.2):
        fail("episode levels out of range")
    flows = axis_flows(1.3, 0.2)
    path = Path(tempfile.gettempdir()) / "check_flows.csv"
    horizon = 40_000
    generate_demand(path, 77, horizon, flow_by_arm=flows)
    with open(path) as stream:
        rows = list(csv.reader(stream))[1:]
    seconds = horizon * 0.2
    for index, arm in enumerate(ARMS):
        count = sum(1 for row in rows if int(row[1]) == index)
        measured = count / seconds * 3600
        print(f"  {arm}: target {flows[arm]:.0f} veh/h, generated {measured:.0f}")
        if abs(measured / flows[arm] - 1) > 0.04:
            fail(f"arm {arm} flow off target")
    if not flows["NW"] > flows["NE"]:
        fail("NW-SE must be the busier axis")
    print("OK demand levels")


def run(run_dir):
    run_dir = Path(run_dir)
    import torch
    manifest = json.loads((run_dir / "demand_manifest.json").read_text())
    with_level = [v for v in manifest.values() if "scale" in v]
    print(f"  demand manifest: {len(manifest)} seeds, {len(with_level)} with a level")
    if not with_level or len(with_level) != len(manifest):
        fail("demand levels missing from the manifest")
    rows = list(csv.reader(open(run_dir / "validation_log.csv")))
    print(f"  validation rows: {len(rows)}, columns {len(rows[0]) if rows else 0}")
    if not rows or len(rows[0]) != 6:
        fail("validation_log must hold episode, step, 3 level scores, score")
    best = torch.load(run_dir / "best_weights.pt", map_location="cpu", weights_only=False)
    settings = best.get("reward_settings", {})
    print(f"  best_weights: reward_settings {settings}, atoms {best.get('atoms')}, v_min {best.get('v_min')}")
    if settings.get("queue_feature_scale") != "log" or best.get("atoms") != 201:
        fail("checkpoint does not carry the observation scale or the C51 support")
    print("OK run files")


if __name__ == "__main__":
    command, rest = sys.argv[1], sys.argv[2:]
    {"native": native, "simhash": simhash, "compare": compare, "config": config,
     "demand": demand, "run": run}[command](*rest)
