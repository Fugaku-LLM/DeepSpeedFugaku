#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM -L elapse=24:00:00
#PJM -L "node=1024"
#PJM --mpi "proc=1024"
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

# Tokenizer setting
CODE_VOCAB_SIZE=20
EN_VOCAB_SIZE=40
JA_VOCAB_SIZE=60

# dataset setting
JA_PERTCENT=30
EN_PERTCENT=65
CODE_PERTCENT=5

# dataset weight setting
ja_wiki_weight=0.005658362414
en_wiki_weight=0.0186473542
ja_cc_weight=0.007745832568
en_pile_weight=0.0140300588
code_stack_weight=0.00625

# training setting
train_token_in_billion=285.2
train_token=$(($train_token_in_billion * 1000 * 1000 * 1000))

lr_warmup_tokens_in_billion=2.852
lr_warmup_tokens=$(($lr_warmup_tokens_in_billion * 1000 * 1000 * 1000))

# same as megatron deepspeed setting
lr_decay_tokens_in_billion=${train_token_in_billion}
lr_decay_tokens=${train_token}

# Change for multinode config
CPUS_PER_NODE=1
NNODES=1024
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE * $NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
CHECKPOINT_PATH=/data/hp190122/share/fujii/checkpoints/llm-jp-v1/1.3b_tp4_dp256_v2.1_code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k/ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}
VOCAB_FILE=tokenizer/models/ver2/code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k.ver2.1.model

mkdir -p $CHECKPOINT_PATH

# dataset setting
DATASET_PATH=/data/hp190122/share/dataset/wikipedia/binarized/v2_1-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k

DATA_PATH=""

DATA_PATH="${DATA_PATH} ${ja_wiki_weight} ${DATASET_PATH}/ja_wiki_text_document" # wiki (ja)

for i in {0..44}; do
  # pile (en)
  DATA_PATH="${DATA_PATH} ${en_pile_weight} ${DATASET_PATH}/en_pile${i}_text_document"
done

DATA_PATH="${DATA_PATH} ${en_wiki_weight} ${DATASET_PATH}/en_wiki_text_document" # wiki (en)

for i in {0..37}; do
  DATA_PATH="${DATA_PATH} ${ja_cc_weight} ${DATASET_PATH}/ja_cc${i}_text_document" # cc (ja)
done

# stack (code)
for i in {0..7}; do
  DATA_PATH="${DATA_PATH} ${code_stack_weight} ${DATASET_PATH}/code_stack${i}_text_document"
done

TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

# distributed setting
output_path="jobs/mpi_outs/${PJM_JOBID}_n${nodos}"
DISTRIBUTED_ARGS="-np $NNODES -std-proc ${output_path}/stdproc"

DATA_PARALLEL_SIZE=256
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
  --num-attention-heads 16 \
  --micro-batch-size 2 \
  --global-batch-size 512 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --train-tokens $train_token \
  --lr-decay-tokens $lr_decay_tokens \
  --lr-warmup-tokens $lr_warmup_tokens \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $DATA_PATH \
  --tokenizer-type JapaneseSentencePiece \
  --vocab-file $VOCAB_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend mpi \
  --init-method-std 0.013 \
  --lr 2.0e-4 \
  --min-lr 1.0e-6 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 1 \
  --save-interval 100 \
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
  --log-optimizer-states-to-tensorboard \
  --wandb-name "1.3b_gb512-ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}"
