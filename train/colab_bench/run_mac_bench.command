#!/bin/zsh -l
# Mac: build the native simulator into train/venv, then a 10-minute training
# speed test with the settings of the long run (6 workers, learner thread).
cd "$(dirname "$0")/../.." || exit 1
STAMP=$(date +%Y%m%d_%H%M)
OUT="train/results/mac_bench_$STAMP"; mkdir -p "$OUT"
LOG="$OUT/console.log"
exec > >(tee -a "$LOG") 2>&1
PY=train/venv/bin/python
echo "[mac] $(date '+%F %T') $(sysctl -n machdep.cpu.brand_string) $(sysctl -n hw.physicalcpu)/$(sysctl -n hw.logicalcpu) cores"
$PY -m pip install -q "Cython==3.3.1" setuptools 2>&1 | tail -2
$PY python-simulation/scripts/build_native.py 2>&1 | tail -1
$PY -c "import sys; sys.path.insert(0, 'python-simulation/simulation'); import python_simulation.python_sim as p; print('native simulator:', p.__file__.endswith('.so'))"
caffeinate -i $PY -u -m train.logic.train --scenario lvl --backend python --algorithm rainbow \
  --reward-mode r1 --workers ${WORKERS:-6} --rollout-mode async --learner thread \
  --total-steps 5000000 --warmup 2000 --validate-every 250 --checkpoint-every 500 \
  --max-train-seconds 600 --output-dir "$OUT/run"
$PY - "$OUT/run" <<'PY'
import json, sys
d = json.load(open(sys.argv[1] + "/training_summary.json"))
keys = ("decisions", "updates", "decisions_per_second", "decisions_per_second_rollout_phase",
        "learner_busy_seconds", "rollouts_held_back_seconds")
print("[mac] summary", {k: d.get(k) for k in keys})
PY
echo "[mac] done $(date '+%F %T')"
