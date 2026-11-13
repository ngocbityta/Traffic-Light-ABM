#!/bin/bash
# =============================================================================
# Setup Python environment cho DQN traffic controller (Phuong an B: GAMA + PyTorch)
# =============================================================================
# Vi sao dung Python 3.11?
#   Moi truong Python 3.11 da duoc kiem thu voi PyTorch va gama-client trong
#   du an; Python 3.14 tren may hien tai gap loi event loop cua gama-client.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "[1/3] Tao virtualenv tai $BASE_DIR/venv (dung $PYTHON_BIN)"
"$PYTHON_BIN" -m venv "$BASE_DIR/venv"

echo "[2/3] Nang cap pip"
"$BASE_DIR/venv/bin/pip" install --quiet --upgrade pip

echo "[3/3] Cai dat dependencies (torch, gama-client, numpy, pandas, matplotlib)"
"$BASE_DIR/venv/bin/pip" install -r "$BASE_DIR/requirements.txt"

"$BASE_DIR/venv/bin/python" - <<'EOF'
import torch, gama_client, numpy
print("VERIFY OK:")
print("  torch       =", torch.__version__)
print("  gama_client =", gama_client.__file__)
print("  numpy       =", numpy.__version__)
EOF
echo "Hoan tat. Kich hoat bang: source $BASE_DIR/venv/bin/activate"
