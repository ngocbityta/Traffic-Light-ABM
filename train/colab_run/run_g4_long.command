#!/bin/zsh -l
# Long training session on a Colab G4 runtime, backed up for resuming anywhere.
#
#  * Continues train/results/$RUN_NAME from this Mac when it holds a
#    checkpoint, else the Google Drive backup, else starts the run.
#  * On Colab the run is copied to Google Drive every SYNC_MINUTES (needs one
#    consent click in the browser when the session starts) and downloaded to
#    train/results/$RUN_NAME.g4_snapshots/latest.tar every DOWNLOAD_MINUTES.
#  * Training stops after G4_HOURS; the final state is installed into
#    train/results/$RUN_NAME and the runtime is stopped (also on errors).
# If the session dies early (out of compute units), run
# run_mac_continue.command: it picks the newest backup and continues here.
[ -f ~/.zshrc ] && source ~/.zshrc >/dev/null 2>&1
export PATH="$HOME/Library/Python/3.14/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export COPYFILE_DISABLE=1
setopt null_glob
cd "$(dirname "$0")/../.." || exit 1
source train/colab_run/run_config.sh
RUN_DIR="train/results/$RUN_NAME"
SNAP="train/results/$RUN_NAME.g4_snapshots"; mkdir -p "$SNAP"
LOG="$SNAP/launcher_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "[g4] $(date '+%F %T') run $RUN_NAME, ${G4_HOURS} h, $G4_WORKERS workers, $(git rev-parse --short HEAD)"
git status --short
colab usage 2>&1 | grep -i "balance\|rate"
tar czf "$SNAP/src.tgz" --exclude=__pycache__ --exclude=.pytest_cache \
  python-simulation/simulation python-simulation/scripts train/__init__.py train/logic train/colab_run
S="gamarun$RANDOM"
colab new --gpu G4 -s "$S" || { echo "[g4] colab new failed"; exit 1; }
trap 'echo "[g4] stopping $S"; colab stop -s "$S"; colab usage 2>&1 | grep -i "balance\|rate"' EXIT
colab upload -s "$S" "$SNAP/src.tgz" /content/gama_src.tgz || exit 1
if [ -f "$RUN_DIR/training_state.pt" ]; then
  echo "[g4] uploading the Mac copy of $RUN_DIR"
  tar cf "$SNAP/upload.tar" -C "$RUN_DIR" --exclude=demand --exclude='*.copying' .
  colab upload -s "$S" "$SNAP/upload.tar" /content/resume_run.tar || exit 1
  rm -f "$SNAP/upload.tar"
fi
echo "[g4] Google Drive: approve access in the browser window that opens (backups every ${SYNC_MINUTES} min)"
colab drivemount -s "$S" || echo "[g4] WARNING: Drive not mounted; backups go to this Mac only"
colab exec -s "$S" -f train/colab_run/start_long.py || exit 1

download() {
  colab download -s "$S" /content/run_snapshot.tar "$SNAP/latest.tar.part" >/dev/null 2>&1 \
    && mv "$SNAP/latest.tar.part" "$SNAP/latest.tar" && echo "[g4] $(date +%T) backup -> $SNAP/latest.tar"
}
END=$((SECONDS + ${G4_HOURS%.*} * 3600 + 7200))
NEXT_DOWNLOAD=$((SECONDS + DOWNLOAD_MINUTES * 60))
while (( SECONDS < END )); do
  sleep 300
  OUTPUT=$(colab exec -s "$S" -f train/colab_run/poll_long.py 2>&1 | grep -v "new version\|colab update\|enable_update_check\|^\[colab\] Run\|^$")
  echo "---- $(date +%T)"; echo "$OUTPUT"
  [[ "$OUTPUT" == *LONG_DONE* ]] && break
  if (( SECONDS >= NEXT_DOWNLOAD )); then download; NEXT_DOWNLOAD=$((SECONDS + DOWNLOAD_MINUTES * 60)); fi
done
download
if [ -f "$SNAP/latest.tar" ]; then
  train/venv/bin/python train/colab_run/pick_latest.py --install "$RUN_DIR" "$SNAP/latest.tar"
fi
echo "[g4] done $(date '+%F %T')"
