#!/usr/bin/env bash
# One-command launcher for the TinyTinyZero demo.
# Usage:   ./run.sh
# First run will create a venv, install deps, and download Qwen2.5-0.5B from HF (~1GB).
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
if [ ! -d "$VENV" ]; then
  echo "[setup] creating venv at $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo ""
echo "[launch] starting Flask on http://127.0.0.1:5055"
echo "         open that URL in your browser and click 'Load Qwen2.5-0.5B'"
echo ""
python app.py
