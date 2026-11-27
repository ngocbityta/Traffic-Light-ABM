#!/bin/zsh -l
# Full pre-flight check on this machine before the 5M run (about 25-40 min).
# Spends no Colab compute units. Log: train/results/_check/check.log
setopt null_glob
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
PY=$REPO/train/venv/bin/python
H=train/colab_run/check_helpers.py
C=train/results/_check
rm -rf "$C"; mkdir -p "$C"
exec > >(tee -a "$C/check.log") 2>&1
source train/colab_run/run_config.sh
RESULTS=()
step() {   # step NAME COMMAND...: run, record PASS/FAIL
  local name=$1; shift
  echo; echo "==== $name ($(date +%T))"
  if "$@"; then RESULTS+=("PASS  $name"); else RESULTS+=("FAIL  $name"); fi
}

echo "[check] $(date '+%F %T') $(uname -sm), $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "[check] git $(git rev-parse --short HEAD 2>/dev/null)"; git status --short
df -h . | tail -1

env_check() {
  [ -x "$PY" ] || { echo "missing $PY: run bash train/scripts/setup_env.sh"; return 1; }
  $PY -m pip install -q "Cython==3.3.1" setuptools pytest 2>&1 | tail -1
  $PY -c "import sys, torch, numpy, Cython; print('python', sys.version.split()[0], '| torch', torch.__version__, '| numpy', numpy.__version__, '| Cython', Cython.__version__)"
}
step "Python environment" env_check
step "Build native simulator" zsh -c "$PY python-simulation/scripts/build_native.py 2>&1 | tail -1 | grep -q 'Native simulator is ready'"
step "Native extensions are loaded" $PY $H native
step "Training settings (run_config.sh)" $PY $H config "${TRAIN_ARGS[@]}"
step "Demand levels 50-130 %, NW-SE busier" $PY $H demand

native_vs_pure() {
  local seed level
  for seed level in 1001 1.0 1002 1.3 1003 0.5; do
    $PY $H simhash "$C/n_$seed.json" $seed $level 0.2 3000 || return 1
    LVL_PURE_PYTHON=1 $PY $H simhash "$C/p_$seed.json" $seed $level 0.2 3000 || return 1
    $PY $H compare "$C/n_$seed.json" "$C/p_$seed.json" || return 1
  done
}
step "Native == pure Python (3 seeds x 3000 ticks)" native_vs_pure

colab_hash() {   # the code copy sent to Colab must hash like this tree
  local tmp=$C/colab_copy
  mkdir -p "$tmp"
  COPYFILE_DISABLE=1 tar czf "$C/src.tgz" --exclude=__pycache__ --exclude=.pytest_cache \
    python-simulation/simulation python-simulation/scripts train/__init__.py train/logic train/colab_run
  tar xzf "$C/src.tgz" -C "$tmp"
  local here there
  here=$($PY $H config "${TRAIN_ARGS[@]}" | grep combined)
  there=$(cd "$tmp" && $PY train/colab_run/check_helpers.py config "${TRAIN_ARGS[@]}" | grep combined)
  echo "  this tree : $here"; echo "  Colab copy: $there"
  [[ -n "$here" && "$here" == "$there" ]]
}
step "Colab code copy has the same source hash" colab_hash

step "Train tests" $PY -m pytest -q -p no:cacheprovider train/tests
step "Simulator tests" zsh -c "cd python-simulation && $PY -m pytest -q -p no:cacheprovider"

sync_resume() {   # short run, SIGKILL, newest backup, resume with fewer workers
  zsh train/colab_run/test_sync_resume.command > "$C/sync_resume.out" 2>&1
  tail -4 "$C/sync_resume.out"
  grep -q "\[test\] PASS" "$C/sync_resume.out" || return 1
  $PY $H run train/results/_sync_test/mac_run
}
step "Kill -9, back up, resume (real TRAIN_ARGS)" sync_resume

evaluate_levels() {
  local level
  for level in 0.5 1.3; do
    $PY -m train.logic.evaluate --weights train/results/_sync_test/mac_run/best_weights.pt \
      --episode-horizon 1200 --episodes 1 --demand-scale $level --axis-imbalance 0.1 \
      --output "$C/eval_$level/metrics.csv" 2>&1 | grep "\[Eval\] Ep" || return 1
  done
}
step "Evaluate best_weights at 50 % and 130 %" evaluate_levels

colab_cli() {
  command -v colab >/dev/null || { echo "colab CLI not found"; return 1; }
  colab usage 2>&1 | grep -i "balance\|rate\|subscription" | head -3
}
step "Colab CLI (no compute units used)" colab_cli

echo; echo "==================== SUMMARY ($(date +%T))"
print -l -- $RESULTS
FAILED=(${(M)RESULTS:#FAIL*})
if (( ${#FAILED} == 0 )); then echo "ALL CHECKS PASSED"; else echo "${#FAILED} CHECK(S) FAILED"; fi
echo "log: $C/check.log"
