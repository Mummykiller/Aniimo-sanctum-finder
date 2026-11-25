#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
LOG_FILE="$DIR/launcher.log"
PYTHON_BIN="$DIR/.venv/bin/python"
DEPS_STAMP="$DIR/.venv/.deps_installed"

if [ -f "$LOG_FILE" ] && [ "$(wc -c < "$LOG_FILE")" -gt 1048576 ]; then
  mv -f "$LOG_FILE" "$LOG_FILE.1"
fi

if [ ! -f "$DEPS_STAMP" ] || [ ! -x "$PYTHON_BIN" ]; then
  if [ ! -x "$PYTHON_BIN" ]; then
    if ! python3 -m venv "$DIR/.venv" >>"$LOG_FILE" 2>&1; then
      rm -rf "$DIR/.venv"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to create .venv (is python3-venv installed?)" >>"$LOG_FILE"
      exit 1
    fi
  fi
  if ! "$PYTHON_BIN" -m pip install -q -r "$DIR/requirements.txt" >>"$LOG_FILE" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to install dependencies" >>"$LOG_FILE"
    exit 1
  fi
  touch "$DEPS_STAMP"
fi

set +e
"$PYTHON_BIN" "$DIR/main.py" "$@" >>"$LOG_FILE" 2>&1
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch.sh exited with code $rc" >>"$LOG_FILE"
  exit "$rc"
fi
