#!/bin/zsh -l
# Continue (or start) the run on this Mac from the newest copy available:
# train/results/$RUN_NAME, the last G4 download, Google Drive for desktop,
# or a folder downloaded from drive.google.com into ~/Downloads/$RUN_NAME.
setopt null_glob
cd "$(dirname "$0")/../.." || exit 1
source train/colab_run/run_config.sh
PY=train/venv/bin/python
RUN_DIR="train/results/$RUN_NAME"
LOG="train/results/$RUN_NAME.mac_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "[mac] $(date '+%F %T') run $RUN_NAME, $MAC_WORKERS workers, up to $MAC_HOURS h"
CANDIDATES=()
[ -f "train/results/$RUN_NAME.g4_snapshots/latest.tar" ] && CANDIDATES+=("train/results/$RUN_NAME.g4_snapshots/latest.tar")
for d in "$HOME"/Library/CloudStorage/GoogleDrive-*/"My Drive"/$DRIVE_SUBDIR/$RUN_NAME \
         "$HOME/Downloads/$RUN_NAME" ${EXTRA_RESUME_SOURCE:+"$EXTRA_RESUME_SOURCE"}; do
  [ -d "$d" ] && CANDIDATES+=("$d")
done
mkdir -p train/results
$PY train/colab_run/pick_latest.py --install "$RUN_DIR" "${CANDIDATES[@]}"
$PY -m pip install -q "Cython==3.3.1" setuptools 2>&1 | tail -1
$PY python-simulation/scripts/build_native.py 2>&1 | tail -1
if [ -f "$RUN_DIR/training_state.pt" ]; then
  WHERE=(--resume-dir "$RUN_DIR" --output-dir "$RUN_DIR")
else
  [ -d "$RUN_DIR" ] && mv "$RUN_DIR" "$RUN_DIR.stale_$(date +%s)"
  WHERE=(--output-dir "$RUN_DIR")
fi
BUDGET=$($PY -c "print(int(float('$MAC_HOURS') * 3600))")
caffeinate -i $PY -u -m train.logic.train "${TRAIN_ARGS[@]}" --workers "$MAC_WORKERS" \
  --checkpoint-every 50 --max-train-seconds "$BUDGET" "${WHERE[@]}"
echo "[mac] exit $? $(date '+%F %T')"
