#!/bin/bash
# One-click: evaluate every checkpoint of meanflow_l_2, skip those already done,
# auto-refresh the FID/IS line chart after each new ckpt.
#
# Usage:
#   bash eval_all.sh                # use defaults
#   NPROC=4 bash eval_all.sh        # override # of GPUs
#   bash eval_all.sh --min-step 50000   # only evaluate ckpts >= step 50000
#   bash eval_all.sh --dry-run      # print plan, don't run
#
# Pre-downloaded weights (VAE + Inception) are pulled from HDFS on first use
# and cached locally; subsequent runs skip the download/extract step.
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

# HDFS location of pre-downloaded VAE / Inception weights. Override with
# HDFS_MODELS_DIR=hdfs://harunasg/... if running on the SG cluster.
HDFS_MODELS_DIR=${HDFS_MODELS_DIR:-${FR}junkun/data_and_model/open_source}

cd "$(dirname "$0")"

# ---- Prepare offline model caches (idempotent) --------------------------------
VAE_DIR="$HOME/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-ema"
INCEPTION_FILE="$HOME/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"

# VAE: ready iff the snapshots dir contains at least one safetensors file
if compgen -G "$VAE_DIR/snapshots/*/diffusion_pytorch_model.safetensors" > /dev/null; then
    echo "[skip] sd-vae-ft-ema already cached at $VAE_DIR"
else
    echo "[get]  sd-vae-ft-ema from $HDFS_MODELS_DIR/sd-vae-ft-ema.zip"
    mkdir -p "$HOME/.cache/huggingface/hub"
    hdfs dfs -get -t 32 "$HDFS_MODELS_DIR/sd-vae-ft-ema.zip" /tmp/sd-vae-ft-ema.zip
    unzip -q -o /tmp/sd-vae-ft-ema.zip -d "$HOME/.cache/huggingface/hub"
    rm -f /tmp/sd-vae-ft-ema.zip
    echo "[done] sd-vae-ft-ema -> $VAE_DIR"
fi

# Inception (torch_fidelity): single .pth file
if [[ -s "$INCEPTION_FILE" ]]; then
    echo "[skip] inception weights already cached at $INCEPTION_FILE"
else
    echo "[get]  inception weights from $HDFS_MODELS_DIR/weights-inception-2015-12-05-6726825d.pth"
    mkdir -p "$(dirname "$INCEPTION_FILE")"
    hdfs dfs -get -t 32 \
        "$HDFS_MODELS_DIR/weights-inception-2015-12-05-6726825d.pth" \
        "$INCEPTION_FILE"
    echo "[done] inception -> $INCEPTION_FILE"
fi

# Force offline mode so HF/transformers never reach the network even if available.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# -------------------------------------------------------------------------------

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
