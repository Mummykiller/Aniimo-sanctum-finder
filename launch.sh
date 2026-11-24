#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$DIR/launcher.log"
PYTHON_BIN="$DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv "$DIR/.venv" >>"$LOG_FILE" 2>&1
  "$PYTHON_BIN" -m pip install -q -r "$DIR/requirements.txt" >>"$LOG_FILE" 2>&1
fi

if ! nice -n 10 "$PYTHON_BIN" "$DIR/main.py" "$@" >>"$LOG_FILE" 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch.sh exited with error" >>"$LOG_FILE"
  exit 1
fi
