#!/bin/bash
# One-click: evaluate every checkpoint of meanflow_l_2, skip those already done,
# auto-refresh the FID/IS line chart after each new ckpt.
#
# Usage:
#   bash eval_all.sh                # use defaults
#   NPROC=4 bash eval_all.sh        # override # of GPUs
#   bash eval_all.sh --min-step 50000   # only evaluate ckpts >= step 50000
#   bash eval_all.sh --dry-run      # print plan, don't run
set -euo pipefail

EXP_NAME=${EXP_NAME:-meanflow_l_2}
OUTPUT_DIR=${OUTPUT_DIR:-exp}
MODEL=${MODEL:-SiT-L/2}
RESOLUTION=${RESOLUTION:-256}
# meanflow_l_2 was trained with CFG (cfg-omega=0.2) -> per README, eval cfg-scale must be 1.0.
CFG_SCALE=${CFG_SCALE:-1.0}
NUM_STEPS=${NUM_STEPS:-1}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-50000}
PER_PROC_BATCH=${PER_PROC_BATCH:-128}
NPROC=${NPROC:-${ARNOLD_WORKER_GPU:-8}}
FID_STATS=${FID_STATS:-./fid_stats/adm_in256_stats.npz}

cd "$(dirname "$0")"

python eval_all.py \
    --exp-name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --resolution "$RESOLUTION" \
    --cfg-scale "$CFG_SCALE" \
    --num-steps "$NUM_STEPS" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --per-proc-batch-size "$PER_PROC_BATCH" \
    --nproc-per-node "$NPROC" \
    --fid-statistics-file "$FID_STATS" \
    "$@"
