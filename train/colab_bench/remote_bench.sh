#!/bin/bash
# Runs on the Colab VM (started by start_bench.py). Writes /content/bench/.
set -u
cd /content && rm -rf gama && mkdir -p gama && tar xzf gama_src.tgz -C gama && cd gama
echo "== machine $(date +%T)"
nproc; lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\)|Core\(s\)|Socket"; free -g | head -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -V; python3 -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"
echo "== setup $(date +%T)"
pip install -q gama-client "Cython==3.3.1" setuptools 2>&1 | tail -2
python3 python-simulation/scripts/build_native.py 2>&1 | tail -1
python3 -c "import sys; sys.path.insert(0, 'python-simulation/simulation'); import python_simulation.python_sim as p; print('native simulator:', p.__file__.endswith('.so'))"
mkdir -p results
CPUS=$(nproc)
run() {  # name workers extra-args...
  local name=$1 workers=$2; shift 2
  if (( workers + 2 > CPUS )); then echo "skip $name: $workers workers on $CPUS CPUs"; return; fi
  echo "== run $name ($workers workers) $(date +%T)"
  python3 -u -m train.logic.train --scenario lvl --backend python --algorithm rainbow \
    --reward-mode r1 --device cuda --workers "$workers" --rollout-mode async \
    --rollout-decisions 16 --learner-threads 4 --episode-horizon 8000 \
    --total-steps 5000000 --max-train-seconds "${BENCH_SECONDS:-600}" --warmup 2000 \
    --validate-every 250 --checkpoint-every 500 \
    --output-dir "results/$name" "$@" > "results/$name.console.log" 2>&1
  echo "exit $? $(date +%T)"
  python3 - "$name" <<'PY'
import json, sys
d = json.load(open(f"results/{sys.argv[1]}/training_summary.json"))
keys = ("decisions", "updates", "decisions_per_second", "decisions_per_second_rollout_phase",
        "rollout_phase_seconds", "learner_busy_seconds", "rollouts_held_back_seconds")
print("summary", sys.argv[1], {k: d.get(k) for k in keys})
PY
}
run g4_46w_thread 46 --learner thread
run g4_40w_thread 40 --learner thread
echo "BENCH_DONE $(date +%T)"
