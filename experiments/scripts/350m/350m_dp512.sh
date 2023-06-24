#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=large"
#PJM -L elapse=24:00:00
#PJM -L "node=512"
#PJM --mpi "proc=512"
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

# Change for multinode config
CPUS_PER_NODE=1
NNODES=512
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE * $NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
CHECKPOINT_PATH=checkpoints/350m_dp512_fp32/
INPUT_PREFIX=dataset
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=data/wikipedia/binarized/gpt-2/ja_wiki_text_document
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

mkdir -p $CHECKPOINT_PATH

output_path="jobs/mpi_outs/${PJM_JOBID}_n${nodos}"
DISTRIBUTED_ARGS="-np $NNODES -std-proc ${output_path}/stdproc"
DATA_PARALLEL_SIZE=512
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
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
  --global-batch-size 512 \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --train-iters 500000 \
  --lr-decay-iters 320000 \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $DATA_PATH \
  --vocab-file $INPUT_PREFIX/$VOCAB_FILE \
  --merge-file $INPUT_PREFIX/$MERGE_FILE \
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
  --save-interval 300 \
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
  --wandb-name "ja-wiki-350m_dp512_fp32"
