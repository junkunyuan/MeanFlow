#!/bin/bash
LOCAL_DST="/opt/tiger/MeanFlow/data_and_model/imagenet_train_latents.lmdb"
HDFS_SRC=${SG}junkun/data_and_model/open_source/ILSVRC/imagenet-1k/data_MeanFlow/imagenet_train_latents.lmdb

if [[ -e "$LOCAL_DST" ]]; then
  echo "ImageNet 已存在,跳过下载: $LOCAL_DST"
else
  mkdir -p /opt/tiger/MeanFlow/data_and_model
  hdfs dfs -get -t 1024 "$HDFS_SRC" "$LOCAL_DST"
  echo "完成ImageNet拷贝"
fi

NNODES=$ARNOLD_NUM
NODE_RANK=$ARNOLD_ID
NPROC_PER_NODE=$ARNOLD_WORKER_GPU
MASTER_ADDRESS=$ARNOLD_WORKER_0_HOST
MASTER_PORT=$PORT0
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

# 要恢复训练就设置为对应的 step（例如 RESUME_STEP=10000），从头训练保持 0
RESUME_STEP=0

accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDRESS \
    --main_process_port $MASTER_PORT \
    train.py \
    --resume-step $RESUME_STEP \
    --exp-name "meanflow_l_2" \
    --output-dir "exp" \
    --data-dir ${LOCAL_DST} \
    --model "SiT-L/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 0.2 \
    --cfg-kappa 0.92 \
    --cfg-min-t 0.0 \
    --cfg-max-t 0.8 \
    --checkpointing-steps 10000
