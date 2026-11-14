#!/usr/bin/env bash
set -euo pipefail
: "${ASSIGNMENT_DIR:?Set to the assignment-2 directory}"
: "${RUN_ROOT:?}"
: "${CAMPAIGN_ROOT:?}"
: "${PARTITIONS_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?Submit indices 0--83}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$CAMPAIGN_ROOT/matplotlib-cache"
"$RUN_ROOT/venv/bin/python" "$ASSIGNMENT_DIR/final_campaign.py" \
  --data-dir "$RUN_ROOT/data" --partitions-root "$PARTITIONS_ROOT" \
  --output "$CAMPAIGN_ROOT" --run-index "$SLURM_ARRAY_TASK_ID"
