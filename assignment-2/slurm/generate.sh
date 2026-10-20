#!/usr/bin/env bash
set -euo pipefail
: "${ASSIGNMENT_DIR:?}"
: "${RUN_ROOT:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$RUN_ROOT/matplotlib-cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$RUN_ROOT/venv/bin/python" "$ASSIGNMENT_DIR/generate_data.py" --output "$RUN_ROOT/data" --device cuda
