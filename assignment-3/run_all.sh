#!/usr/bin/env bash
# Full pipeline. Set NSHARDS to the number of CPU cores you want to use.
set -euo pipefail
export KERAS_BACKEND=jax
NSHARDS=${NSHARDS:-2}
mkdir -p results logs

python data.py

# ---- Task A: grids ---------------------------------------------------------
python sweep.py --mode nondp

sharded () {            # sharded <mode> [extra args...]; merges into results/<mode>.csv
  local mode=$1; shift
  for s in $(seq 0 $((NSHARDS-1))); do
    python sweep.py --mode "$mode" --shard "$s" --nshards "$NSHARDS" "$@" > "logs/$mode.shard$s.log" 2>&1 &
  done
  wait
  python - "$mode" <<'EOF'
import glob, sys, pandas as pd
m = sys.argv[1]
parts = [pd.read_csv(f) for f in sorted(glob.glob(f"results/{m}.shard*.csv"))]
if parts:
    pd.concat(parts).drop_duplicates("run_id").to_csv(f"results/{m}.csv", index=False)
EOF
}

# Pre-compute noise multipliers once so shards don't race on the cache file.
python - <<'EOF'
import itertools, common as C
n = len(C.load_data()["X_train"])
for e, B in itertools.product(C.EPSILONS, C.BATCH_SIZES):
    C.noise_multiplier(e, B, C.steps_for(B), n)
EOF
sharded dp

# ---- Task B: audit (also writes results/selected_config.json) ---------------
python audit.py

# ---- Extensions / ablations at the selected configuration -------------------
read B LR NB NLR < <(python -c "import json;s=json.load(open('results/selected_config.json'));print(s['dp_B'],s['dp_lr'],s['nondp_B'],s['nondp_lr'])")
sharded lowlr
sharded steps --batch "$B" --lr "$LR" --nondp_batch "$NB" --nondp_lr "$NLR"
sharded clip  --batch "$B" --lr "$LR"
sharded seeds --batch "$B" --lr "$LR" --nondp_batch "$NB" --nondp_lr "$NLR"
python audit.py --canaries --design ood
python audit.py --canaries --design mislabel
python accounting_check.py

# ---- Figures and report tables -----------------------------------------------
python make_figures.py
echo "done: results/*.csv, figures/*.png, results/latex/*.tex"
