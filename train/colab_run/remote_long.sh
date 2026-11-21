#!/bin/bash
# Colab side of run_g4_long.command. Started in the background by start_long.py.
set -u
cd /content && rm -rf gama && mkdir -p gama && tar xzf gama_src.tgz -C gama && cd gama
source train/colab_run/run_config.sh
RUN_DIR="/content/runs/$RUN_NAME"
DRIVE_DIR="/content/drive/MyDrive/$DRIVE_SUBDIR/$RUN_NAME"
mkdir -p /content/runs
echo "== machine $(date +%T)"; nproc; nvidia-smi --query-gpu=name --format=csv,noheader
pip install -q gama-client "Cython==3.3.1" setuptools 2>&1 | tail -2
python3 python-simulation/scripts/build_native.py 2>&1 | tail -1

# Where this session continues from: the copy uploaded from the Mac, else
# the Google Drive backup, else a new run.
if [ -f /content/resume_run.tar ]; then
  rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"; tar xf /content/resume_run.tar -C "$RUN_DIR"
  echo "== resume from the Mac copy"
elif [ -f "$DRIVE_DIR/training_state.pt" ]; then
  rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
  find "$DRIVE_DIR" -maxdepth 1 -type f ! -name '*.copying' -exec cp -p {} "$RUN_DIR"/ \;
  echo "== resume from Google Drive"
else
  if [ -d "$RUN_DIR" ]; then mv "$RUN_DIR" "$RUN_DIR.stale_$(date +%s)"; fi
  echo "== new run"
fi
if [ -f "$RUN_DIR/training_state.pt" ]; then
  WHERE=(--resume-dir "$RUN_DIR" --output-dir "$RUN_DIR")
else
  WHERE=(--output-dir "$RUN_DIR")
fi
mkdir -p "$RUN_DIR"
if [ -d /content/drive/MyDrive ]; then
  mkdir -p "$DRIVE_DIR"; echo "== Google Drive backup every ${SYNC_MINUTES} min: $DRIVE_DIR"
else
  echo "== WARNING: Google Drive not mounted; backups only reach the Mac (every ${DOWNLOAD_MINUTES} min)"
fi
python3 train/colab_run/sync_run.py --src "$RUN_DIR" --dst "$DRIVE_DIR" \
  --tar /content/run_snapshot.tar --every $((SYNC_MINUTES * 60)) --until /content/TRAIN_DONE \
  > /content/sync.log 2>&1 &
SYNC_PID=$!
SECONDS_BUDGET=$(python3 -c "print(int(float('$G4_HOURS') * 3600))")
echo "== train $(date +%T): $G4_WORKERS workers, ${SECONDS_BUDGET}s"
python3 -u -m train.logic.train "${TRAIN_ARGS[@]}" --device "$G4_DEVICE" --workers "$G4_WORKERS" \
  --learner-threads 4 --checkpoint-every 500 --max-train-seconds "$SECONDS_BUDGET" "${WHERE[@]}" \
  >> "$RUN_DIR/console.log" 2>&1
echo "== train exit $? $(date +%T)"
touch /content/TRAIN_DONE
wait $SYNC_PID
tail -2 /content/sync.log
echo "LONG_DONE $(date +%T)"
