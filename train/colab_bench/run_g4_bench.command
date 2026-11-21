#!/bin/zsh -l
# Mac launcher: 10-minute training benchmarks on a Colab G4 runtime.
# The runtime is stopped when this script ends (trap), success or not.
[ -f ~/.zshrc ] && source ~/.zshrc >/dev/null 2>&1
export PATH="$HOME/Library/Python/3.14/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export COPYFILE_DISABLE=1
cd "$(dirname "$0")/../.." || exit 1
STAMP=$(date +%Y%m%d_%H%M)
OUT="train/results/g4_bench_$STAMP"; mkdir -p "$OUT"
LOG="$OUT/launcher.log"
exec > >(tee -a "$LOG") 2>&1
echo "[launcher] $(date '+%F %T') $(git rev-parse --short HEAD) + working tree"
git status --short
S="gamabench$RANDOM"
tar czf "$OUT/src.tgz" --exclude=__pycache__ --exclude=.pytest_cache \
  python-simulation/simulation python-simulation/scripts train/__init__.py train/logic train/colab_bench
colab new --gpu G4 -s "$S" || { echo "[launcher] colab new failed"; exit 1; }
trap 'echo "[launcher] stopping $S"; colab stop -s "$S"; colab usage' EXIT
colab upload -s "$S" "$OUT/src.tgz" /content/gama_src.tgz || exit 1
echo 'import subprocess; subprocess.run("mkdir -p /content/gama_src && tar xzf /content/gama_src.tgz -C /content/gama_src", shell=True, check=True); print("unpacked")' | colab exec -s "$S" || exit 1
colab exec -s "$S" -f train/colab_bench/start_bench.py || exit 1
END=$((SECONDS + 3 * 3600))
while (( SECONDS < END )); do
  sleep 60
  OUTPUT=$(colab exec -s "$S" -f train/colab_bench/poll_bench.py 2>&1 | grep -v "new version\|colab update\|enable_update_check\|^\[colab\] Run\|^$")
  echo "---- $(date +%T)"; echo "$OUTPUT"
  [[ "$OUTPUT" == *BENCH_DONE* ]] && break
done
colab exec -s "$S" -f train/colab_bench/pack_bench.py
colab download -s "$S" /content/bench_results.tgz "$OUT/bench_results.tgz" && \
  tar xzf "$OUT/bench_results.tgz" -C "$OUT"
echo "[launcher] done $(date '+%F %T')"
