#!/bin/bash
# Mo UI quan sat nga tu Le Van Luong voi den do Rainbow DQN (R1) da train dieu khien.
# Luu luong 3000 xe/h moi nhanh. Bam dup file nay trong Finder, hoac chay: bash scripts/chay_ui_rainbow_r1.command
cd "$(dirname "$0")/.." || exit 1
PY=train/venv/bin/python
[ -x "$PY" ] || PY=python3
WEIGHTS=train/results/rainbow_r1_5m/best_weights.pt
if [ ! -f "$WEIGHTS" ]; then
  echo "Dang giai nen weights tu rainbow_r1_5m_final.tar.zst ..."
  mkdir -p train/results
  zstd -dc rainbow_r1_5m_final.tar.zst | tar -x -C train/results \
    rainbow_r1_5m/training_config.json rainbow_r1_5m/best_weights.pt || exit 1
fi
echo "Build ban native cho ma mo phong moi (lan dau mat vai phut, loi thi chay ban Python thuan)..."
"$PY" python-simulation/scripts/build_native.py 2>&1 | tail -1
(sleep 4; open "http://127.0.0.1:8765/") &
exec "$PY" scripts/lvl_observer_server.py --flow 3000 --weights "$WEIGHTS" "$@"
