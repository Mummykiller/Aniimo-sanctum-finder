#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$DIR/launcher.log"

if [ -x "$DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if ! nice -n 10 "$PYTHON_BIN" "$DIR/main.py" "$@" >>"$LOG_FILE" 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch.sh exited with error" >>"$LOG_FILE"
  exit 1
fi
