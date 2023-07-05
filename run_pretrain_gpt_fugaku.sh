#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=02:00:00
#PJM -L "node=1"
#PJM --mpi "proc=4"
#PJM --mpi "max-proc-per-node=4"
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin
export HF_DATASETS_CACHE="$HOME/work/DeepSpeedFugaku/.cache"

CHECKPOINT_PATH=checkpoints/pretrain_gpt2/
INPUT_PREFIX=dataset
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=/data/hp190122/gpt-fugaku-data/codeparrot/codeparrot_content_document
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
WORLD_SIZE=4
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
         --nnodes 1 \
         --node_rank 0 \
         --master_addr localhost \
         --master_port 6000"
export OMP_PROC_BIND=FALSE
OMP_NUM_THREADS=4 python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_gpt.py \
        $MODEL_ARGS \
        $OUTPUT_ARGS \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 100 \
    --lr-decay-iters 320000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file $INPUT_PREFIX/$VOCAB_FILE \
    --merge-file $INPUT_PREFIX/$MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend gloo \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --checkpoint-activations \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --DDP-impl torch \
    --no-cuda \
    --no-load-rng \
    $TENSORBOARD_ARGS
