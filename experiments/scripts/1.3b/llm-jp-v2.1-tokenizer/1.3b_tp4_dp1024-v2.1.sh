#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM -L elapse=24:00:00
#PJM -L "node=4096"
#PJM --mpi "proc=4096"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp230254
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin

user_name=$(whoami)
cd /home/$user_name/work/DeepSpeedFugaku

CODE_VOCAB_SIZE=30
EN_VOCAB_SIZE=80
JA_VOCAB_SIZE=120

# Change for multinode config
CPUS_PER_NODE=1
NNODES=4096
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE * $NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
CHECKPOINT_PATH=/data/hp190122/share/fujii/checkpoints/llm-jp-v1/1.3b_tp4_dp1024_v2.1
INPUT_PREFIX=dataset
VOCAB_FILE=tokenizer/models/ver2/code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k.ver2.1.model

mkdir -p $CHECKPOINT_PATH

# dataset setting
DATASET_PATH=/data/hp190122/share/dataset/wikipedia/binarized/v2_1-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k

DATA_PATH="1 ${DATASET_PATH}/code_stack_text_document" # stack (code)

for i in {0..44}; do
  DATA_PATH="${DATA_PATH} 1 ${DATASET_PATH}/en_pile${i}_text_document" # pile (en)
done

DATA_PATH="${DATA_PATH} 1 ${DATASET_PATH}/en_wiki_text_document" # wiki (en)

for i in {0..37}; do
  DATA_PATH="${DATA_PATH} 1 ${DATASET_PATH}/ja_cc${i}_text_document" # cc (ja)
done

DATA_PATH="${DATA_PATH} 1 ${DATASET_PATH}/ja_wiki_text_document" # wiki (ja)

TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

# distributed setting
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
  --hidden-size 1024 \
  --num-attention-heads 16 \
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
  --save-interval 1000 \
  --eval-interval 100 \
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
  --wandb-name "llm-jp-1.3b_tp4_dp1024-v2.1-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k"
