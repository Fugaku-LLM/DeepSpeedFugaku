#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=large"
#PJM -L elapse=24:00:00
#PJM -L "node=4096"
#PJM --mpi "proc=4096"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp190122
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin

user_name=$(whoami)
cd /home/$user_name/work/DeepSpeedFugaku

JA_VOCAB_SIZE=10
EN_VOCAB_SIZE=40

# Change for multinode config
CPUS_PER_NODE=1
NNODES=4096
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE*$NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
CHECKPOINT_PATH=checkpoints/ja-en-wiki/1.3b_mp4_dp1024_v1_ja${JA_VOCAB_SIZE}K_en${EN_VOCAB_SIZE}K
INPUT_PREFIX=dataset
VOCAB_FILE=tokenizer/models/cc100ja1GB_cc100en1GB/cc100_ja${JA_VOCAB_SIZE}K_en${EN_VOCAB_SIZE}K.symbolRemoved.vocab.reestimated.model

DATA_PATH="
  1 data/wikipedia/binarized/v1-ja${JA_VOCAB_SIZE}K-en${EN_VOCAB_SIZE}K/ja_wiki_text_document
  1 data/wikipedia/binarized/v1-ja${JA_VOCAB_SIZE}K-en${EN_VOCAB_SIZE}K/en_wiki_text_document
"

TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

output_path="jobs/mpi_outs/${PJM_JOBID}_n${nodos}"
DISTRIBUTED_ARGS="-np $NNODES -std-proc ${output_path}/stdproc"

DATA_PARALLEL_SIZE=1024

PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=4
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
DATA_PARALLEL_ARGS="--DDP-impl local"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

export OMP_NUM_THREADS=48

mpirun $DISTRIBUTED_ARGS \
  python pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tokenizer-type JapaneseSentencePiece \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend mpi \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --no-cuda \
    --checkpoint-activations \
    --use-cpu-initialization \
    --num-workers 0 \
    --no-load-rng \
    $PARALLEL_ARGS \
    $TENSORBOARD_ARGS \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --wandb-name "ja-en-wiki-1.3b_mp4_dp1024-v1-ja${JA_VOCAB_SIZE}k-env${EN_VOCAB_SIZE}k"
