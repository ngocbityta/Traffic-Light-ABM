#!/bin/zsh -l
# Local rehearsal of "the Colab session dies, continue on the Mac":
# a short run is backed up every 10 s, killed with SIGKILL (no clean
# shutdown, like a runtime that runs out of compute units), then the newest
# backup is installed and training continues with another worker count.
setopt null_glob
cd "$(dirname "$0")/../.." || exit 1
PY=train/venv/bin/python
T=train/results/_sync_test
rm -rf "$T"; mkdir -p "$T"
exec > >(tee -a "$T/test.log") 2>&1
ARGS=(--scenario lvl --backend python --algorithm rainbow --reward-mode r1 --rollout-mode async
      --learner thread --episode-horizon 1200 --total-steps 5000000 --warmup 60 --validate-every 4)
echo "[test] $(date +%T) phase 1: 3 workers, backup every 10 s"
set -m   # the trainer gets its own process group, killed as a whole below
$PY -u -m train.logic.train $ARGS --workers 3 --checkpoint-every 2 --max-train-seconds 600 \
  --output-dir "$T/colab_run" > "$T/phase1.log" 2>&1 &
TRAINER=$!
set +m
$PY train/colab_run/sync_run.py --src "$T/colab_run" --dst "$T/fake_drive" --tar "$T/latest.tar" \
  --every 10 --until "$T/STOP_SYNC" > "$T/sync.log" 2>&1 &
SYNC=$!
sleep 100
kill -9 -- -$TRAINER; echo "[test] $(date +%T) trainer and its workers killed (SIGKILL)"
touch "$T/STOP_SYNC"; wait $SYNC; tail -2 "$T/sync.log"
echo "[test] phase 2: pick the newest backup, continue with 2 workers"
$PY train/colab_run/pick_latest.py --install "$T/mac_run" "$T/latest.tar" "$T/fake_drive"
BEFORE=$($PY -c "import torch;print(torch.load('$T/mac_run/training_state.pt',weights_only=False)['agent']['decisions'])")
$PY -u -m train.logic.train $ARGS --workers 2 --checkpoint-every 3 --max-train-seconds 60 \
  --resume-dir "$T/mac_run" --output-dir "$T/mac_run" > "$T/phase2.log" 2>&1
CODE=$?
grep -v "^\[Rollout" "$T/phase2.log" | head -8
AFTER=$($PY -c "import json;print(json.load(open('$T/mac_run/training_summary.json'))['decisions'])")
echo "[test] resumed at $BEFORE decisions, ended at $AFTER (exit $CODE)"
if [[ $CODE == 0 && $AFTER -gt $BEFORE ]]; then echo "[test] PASS"; else echo "[test] FAIL"; fi
