#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM --rsc-list "proc-openfd=65536"
#PJM -L elapse=24:00:00
#PJM -L "node=1024"
#PJM --mpi "proc=1024"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp230254
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

set -e

# python vertualenv setting
source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin

# project directory setting
user_name=$(whoami)
cd /home/$user_name/work/DeepSpeedFugaku

# distributed setting
# Change for multinode config
CPUS_PER_NODE=1 # fixed (Fugaku)
NNODES=1024
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE * $NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))

# Tokenizer setting
TOKENIZER_PATH=llm-jp-tokenizer/models/ver2.2/code10K_en20K_ja30K.ver2.2.model

# dataset weight setting
JA_PERTCENT=50
EN_PERTCENT=50
CODE_PERTCENT=0

DATA_PARALLEL_SIZE=256
PIPELINE_MODEL_PARALLEL_SIZE=4
TENSOR_MODEL_PARALLEL_SIZE=1

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512

# gradinet accumulation size
if [ $(($MICRO_BATCH_SIZE * $DATA_PARALLEL_SIZE)) -ne $GLOBAL_BATCH_SIZE ]; then
  GRADIENT_ACCUMULATION_STEPS=$(($GLOBAL_BATCH_SIZE / ($MICRO_BATCH_SIZE * $DATA_PARALLEL_SIZE)))
  echo "gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"

  if [ $(($GRADIENT_ACCUMULATION_STEPS * $MICRO_BATCH_SIZE * $DATA_PARALLEL_SIZE)) -ne $GLOBAL_BATCH_SIZE ]; then
    echo "Error: grad_acc * micro_batch * data_parallel_size != global_batch_size"
    echo " grad_acc * micro_batch * data_parallel_size = $GRADIENT_ACCUMULATION_STEPS * $MICRO_BATCH_SIZE * $DATA_PARALLEL_SIZE = $(($GRADIENT_ACCUMULATION_STEPS * $MICRO_BATCH_SIZE * $DATA_PARALLEL_SIZE)) but global_batch_size = $GLOBAL_BATCH_SIZE"
    exit 1
  fi
fi

# checkpoint setting
CHECKPOINT_PATH=/data/hp190122/share/fujii/checkpoints/llm-jp-v1/code10K_en20K_ja30K.ver2.2/1.3b/dp${DATA_PARALLEL_SIZE}_pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}/ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}
mkdir -p $CHECKPOINT_PATH

# dataset setting
DATASET_PATH=/data/hp190122/share/dataset/llm-jp-corpus-v1.0.2/v2_2-code10k_en20k_ja30k

# train data setting
TRAIN_DATA_PATH=""

# japanese wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1593695182 ${DATASET_PATH}/ja_wiki_merged_train_0_text_document"
# english wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5084807913 ${DATASET_PATH}/en_wiki_merged_train_0_text_document"
# japanese cc
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9775284297 ${DATASET_PATH}/ja_cc_merged_train_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9756689602 ${DATASET_PATH}/ja_cc_merged_train_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9786250625 ${DATASET_PATH}/ja_cc_merged_train_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9777791915 ${DATASET_PATH}/ja_cc_merged_train_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9765518836 ${DATASET_PATH}/ja_cc_merged_train_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9769646834 ${DATASET_PATH}/ja_cc_merged_train_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9777598507 ${DATASET_PATH}/ja_cc_merged_train_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9785135661 ${DATASET_PATH}/ja_cc_merged_train_7_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9778664499 ${DATASET_PATH}/ja_cc_merged_train_8_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9620536780 ${DATASET_PATH}/ja_cc_merged_train_9_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9784389547 ${DATASET_PATH}/ja_cc_merged_train_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9779047758 ${DATASET_PATH}/ja_cc_merged_train_11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9781273054 ${DATASET_PATH}/ja_cc_merged_train_12_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9779274519 ${DATASET_PATH}/ja_cc_merged_train_13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9590208758 ${DATASET_PATH}/ja_cc_merged_train_14_text_document"

# english pile
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3390882396 ${DATASET_PATH}/en_pile_merged_train_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3377542797 ${DATASET_PATH}/en_pile_merged_train_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3378903914 ${DATASET_PATH}/en_pile_merged_train_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3411700284 ${DATASET_PATH}/en_pile_merged_train_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3406369579 ${DATASET_PATH}/en_pile_merged_train_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3408234653 ${DATASET_PATH}/en_pile_merged_train_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3393537510 ${DATASET_PATH}/en_pile_merged_train_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3378628450 ${DATASET_PATH}/en_pile_merged_train_7_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3414266377 ${DATASET_PATH}/en_pile_merged_train_8_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3415215745 ${DATASET_PATH}/en_pile_merged_train_9_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3373992692 ${DATASET_PATH}/en_pile_merged_train_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3385585842 ${DATASET_PATH}/en_pile_merged_train_11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3378092975 ${DATASET_PATH}/en_pile_merged_train_12_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3390762623 ${DATASET_PATH}/en_pile_merged_train_13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3372460710 ${DATASET_PATH}/en_pile_merged_train_14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3388037651 ${DATASET_PATH}/en_pile_merged_train_15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3379301168 ${DATASET_PATH}/en_pile_merged_train_16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3398322457 ${DATASET_PATH}/en_pile_merged_train_17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3387856841 ${DATASET_PATH}/en_pile_merged_train_18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3398437994 ${DATASET_PATH}/en_pile_merged_train_19_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3378120478 ${DATASET_PATH}/en_pile_merged_train_20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3324873160 ${DATASET_PATH}/en_pile_merged_train_21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3398887298 ${DATASET_PATH}/en_pile_merged_train_22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3383242566 ${DATASET_PATH}/en_pile_merged_train_23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3376140562 ${DATASET_PATH}/en_pile_merged_train_24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3399056755 ${DATASET_PATH}/en_pile_merged_train_25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3370639761 ${DATASET_PATH}/en_pile_merged_train_26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3390144908 ${DATASET_PATH}/en_pile_merged_train_27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3378417197 ${DATASET_PATH}/en_pile_merged_train_28_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3384487188 ${DATASET_PATH}/en_pile_merged_train_29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3385287883 ${DATASET_PATH}/en_pile_merged_train_30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3382520341 ${DATASET_PATH}/en_pile_merged_train_31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3405202461 ${DATASET_PATH}/en_pile_merged_train_32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3412913401 ${DATASET_PATH}/en_pile_merged_train_33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3425156752 ${DATASET_PATH}/en_pile_merged_train_34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3396089610 ${DATASET_PATH}/en_pile_merged_train_35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3376351472 ${DATASET_PATH}/en_pile_merged_train_36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3386901071 ${DATASET_PATH}/en_pile_merged_train_37_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3382546054 ${DATASET_PATH}/en_pile_merged_train_38_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3385464281 ${DATASET_PATH}/en_pile_merged_train_39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3385837582 ${DATASET_PATH}/en_pile_merged_train_40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3412400983 ${DATASET_PATH}/en_pile_merged_train_41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 467384042 ${DATASET_PATH}/en_pile_merged_train_42_text_document"

# stack (code)
# TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 77377944 ${DATASET_PATH}/code_stack_merged_train_0_text_document"

# validation data setting
VALID_DATA_PATH=""

# japanese wikipedia
VALID_DATA_PATH="${VALID_DATA_PATH} 1095092 ${DATASET_PATH}/ja_wiki_validation_0_text_document"
# english wikipedia
VALID_DATA_PATH="${VALID_DATA_PATH} 1011609 ${DATASET_PATH}/en_wiki_validation_0_text_document"
# japanese cc
VALID_DATA_PATH="${VALID_DATA_PATH} 147265564 ${DATASET_PATH}/ja_cc_validation_0_text_document"
# english pile
VALID_DATA_PATH="${VALID_DATA_PATH} 61267759 ${DATASET_PATH}/en_pile_validation_0_text_document"
# code stack
# VALID_DATA_PATH="${VALID_DATA_PATH} 17449068526 ${DATASET_PATH}/code_stack_merged_train_0_text_document"

# training setting
train_token_in_billion=295.8020127
train_token=$(echo "$train_token_in_billion * 1000 * 1000 * 1000" | bc)
train_token=$(echo "$train_token/1" | bc)

# default megatron-deepspeed confgiraution is 3000 million, but they train model using 300 billion tokens. we use 206 billion tokens, so we set 2.06 billion tokens to lr-warmup-tokens.
lr_warmup_tokens_in_billion=2.95
lr_warmup_tokens=$(echo "$lr_warmup_tokens_in_billion * 1000 * 1000 * 1000" | bc)
lr_warmup_tokens=$(echo "$lr_warmup_tokens/1" | bc)

# same as megatron deepspeed setting
lr_decay_tokens_in_billion=${train_token_in_billion}
lr_decay_tokens=${train_token}

# logging setting
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
# mpi setting
DISTRIBUTED_ARGS="-np $NNODES"

# distributed setting
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
DATA_PARALLEL_ARGS="--DDP-impl local"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

export OMP_NUM_THREADS=48

SEQUENCE_LENGTH=2048
train_samples=$((300 * 1000000000 * 2 / ${SEQUENCE_LENGTH}))

mpirun $DISTRIBUTED_ARGS \
  python pretrain_gpt.py \
  --num-layers 24 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --seq-length $SEQUENCE_LENGTH \
  --max-position-embeddings $SEQUENCE_LENGTH \
  --train-tokens $train_token \
  --train-samples $train_samples \
  --lr-decay-tokens $lr_decay_tokens \
  --lr-warmup-tokens $lr_warmup_tokens \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $TRAIN_DATA_PATH \
  --tokenizer-type JapaneseSentencePiece \
  --vocab-file $TOKENIZER_PATH \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend mpi \
  --init-method-std 0.013 \
  --lr 2.0e-4 \
  --min-lr 1.0e-6 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --cpu-optimizer \
  --cpu-torch-adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 1 \
  --save-interval 500 \
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
  --wandb-entity "gpt-fugaku" \
  --wandb-project "1.3B-llm-jp-dataset-v2.2-tokenizer" \
  --wandb-name "1.3b_dp${DATA_PARALLEL_SIZE}_pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}_gbs${GLOBAL_BATCH_SIZE}_ja${JA_PERTCENT}%_en${EN_PERTCENT}%_code${CODE_PERTCENT}%"
