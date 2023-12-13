#!/bin/bash -x

mkdir -p /local/fcc/pytorch
cd /local/fcc
tar xf /home/u11890/work/1701935794.711074240.fcc.pytorch.y.r1.13_for_a64fx.tar.gz
source /local/fcc/inst/venv/bin/activate
cd /home/u11890/work/training/DeepSpeedFugaku

# distributed setting
# Change for multinode config
CPUS_PER_NODE=1 # fixed (Fugaku)
NNODES=3456
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE * $NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))

# Tokenizer setting
TOKENIZER_PATH=llm-jp-tokenizer/models/ver2.2/code10K_en20K_ja30K.ver2.2.model

# distributed setting
DATA_PARALLEL_SIZE=144
PIPELINE_MODEL_PARALLEL_SIZE=8
TENSOR_MODEL_PARALLEL_SIZE=3

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2304

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
CHECKPOINT_PATH=/data/hp190122/share/takumi/checkpoints/gpt-fugaku-dataset/code10K_en20K_ja30K.ver2.2/13b/dp${DATA_PARALLEL_SIZE}_pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}/gbs${GLOBAL_BATCH_SIZE}
mkdir -p $CHECKPOINT_PATH

# dataset setting
DATASET_PATH=/data/hp190122/share/dataset/fugaku_13b/binarized/v2_2-code10k_en20k_ja30k

# train data setting
TRAIN_DATA_PATH=""

# en books
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 21533774460 ${DATASET_PATH}/books_merged_text_document"
# en arxiv
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14300397157 ${DATASET_PATH}/lumi_en_arxiv_merge_text_document"
# en falcon (refined-web)
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11253894954 ${DATASET_PATH}/lumi_en_falcon_merge_text_document"
# en pile free-law
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11955669467 ${DATASET_PATH}/pile_FreeLaw_merge_text_document"
# en pile pubmed
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5308757461 ${DATASET_PATH}/pile_pubmed_merge_text_document"
# en red-pajama cc
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 31674422340 ${DATASET_PATH}/red_pajama_cc_merge_text_document"
# en red-pajama arxiv
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9562707240 ${DATASET_PATH}/red_pajama_arxiv_merge_text_document"
# en 10_k
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 369623371 ${DATASET_PATH}/10_k_text_document"
# en atticus is contracts and legal
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10696864 ${DATASET_PATH}/atticus_cuad_muad_contracts_text_document"
# en pile philarchive
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 347850794 ${DATASET_PATH}/pile_PhilArchive_text_document"
# en pile nih text
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 422999727 ${DATASET_PATH}/pile_NIH_text_document"
# en parliament
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 133677757 ${DATASET_PATH}/parliament_text_document"
# en climabench
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9218044 ${DATASET_PATH}/climabench_text_document"
# en redpajama-stackexchange
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4602769439 ${DATASET_PATH}/red_pajama_stackexchange_text_document"
# en pile stackexchange
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2588691546 ${DATASET_PATH}/pile_stackexchange_text_document"
# en pile uspto
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6484246795 ${DATASET_PATH}/pile_uspto_merge_text_document"
# en wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5084807913 ${DATASET_PATH}/en_wiki_merged_text_document"

# ja wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1593695182 ${DATASET_PATH}/ja_wiki_merged_text_document"
# cyberagent filtered ja
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 31808623323 ${DATASET_PATH}/ca_filter2ca_cc_filtered_org-bwords_reform_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22976439227 ${DATASET_PATH}/ca_filter2ca_cc2_filtered_org-bwords_text_document"
# okazaki lab cc ja
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14184147835 ${DATASET_PATH}/split_reformat_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14440507382 ${DATASET_PATH}/split_reformat_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13821488987 ${DATASET_PATH}/split_reformat_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16717225516 ${DATASET_PATH}/split_reformat_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16634821784 ${DATASET_PATH}/split_reformat_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22117133216 ${DATASET_PATH}/split_reformat_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 25942389333 ${DATASET_PATH}/split_reformat_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 37326773828 ${DATASET_PATH}/split_reformat_7_text_document"

# math llemma
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11068009210 ${DATASET_PATH}/EleutherAI___proof-pile-2_algebraic-stack_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14680659112 ${DATASET_PATH}/EleutherAI___proof-pile-2_open-web-math_text_document"

# code
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16945303758 ${DATASET_PATH}/markdown_part_merge_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2159043136 ${DATASET_PATH}/rust_part_merge_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 973518264 ${DATASET_PATH}/tex_part_merge_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10966015578 ${DATASET_PATH}/python_part_merge_text_document"

# training setting
train_token_in_billion=400
train_token=$(echo "$train_token_in_billion * 1000 * 1000 * 1000" | bc)
train_token=$(echo "$train_token/1" | bc)

# default megatron-deepspeed confgiraution is 3000 million, but they train model using 300 billion tokens.
lr_warmup_tokens_in_billion=4
lr_warmup_tokens=$(echo "$lr_warmup_tokens_in_billion * 1000 * 1000 * 1000" | bc)
lr_warmup_tokens=$(echo "$lr_warmup_tokens/1" | bc)

# same as megatron deepspeed setting
lr_decay_tokens_in_billion=${train_token_in_billion}
lr_decay_tokens=${train_token}

# logging setting
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

# distributed setting
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
DATA_PARALLEL_ARGS="--DDP-impl local"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

export OMP_NUM_THREADS=48
export LD_PRELOAD=$1
export MYGEMM=99

SEQUENCE_LENGTH=2048
train_samples=$((300 * 1000000000 * 2 / ${SEQUENCE_LENGTH}))

numactl -m 4-7 -N 4-7 \
  python pretrain_gpt.py \
  --num-layers 40 \
  --hidden-size 5184 \
  --num-attention-heads 36 \
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
  --init-method-std 0.008 \
  --lr 1.0e-4 \
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
  --save-interval 250 \
  --eval-interval 100 \
  --eval-iters 10 \
  --no-cuda \
  --checkpoint-activations \
  --use-cpu-initialization \
  --num-workers 1 \
  $PARALLEL_ARGS \
  $TENSORBOARD_ARGS \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --log-timers-to-tensorboard \
  --log-optimizer-states-to-tensorboard \
  --use-flush-denormal \
  --wandb-entity "gpt-fugaku" \
  --wandb-project "13B-2023-12-12" \
  --wandb-name "13b_dp${DATA_PARALLEL_SIZE}_pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}_gbs${GLOBAL_BATCH_SIZE}"
