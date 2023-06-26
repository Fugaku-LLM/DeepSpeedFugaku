#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=12:00:00
#PJM -L "node=384"
#PJM --mpi "proc=384"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp190122          
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin
cd /home/u11078/work/DeepSpeedFugaku

# Change for multinode config
CPUS_PER_NODE=1
NNODES=384
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE*$NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
CHECKPOINT_PATH=checkpoints/30b_pp48_mp8/
INPUT_PREFIX=dataset
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=data/codeparrot/codeparrot_content_document
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

output_path="jobs/mpi_outs/${PJM_JOBID}_n${nodos}"
DISTRIBUTED_ARGS="-np $NNODES -std-proc ${output_path}/stdproc"
DATA_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=48
TENSOR_MODEL_PARALLEL_SIZE=8
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
#DATA_PARALLEL_ARGS="--DDP-impl torch"
#PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

#OMP_PARALLEL_ARGS="OMP_NUM_THREADS=48"
export OMP_NUM_THREADS=48

mpirun $DISTRIBUTED_ARGS \
python pretrain_gpt.py \
    --num-layers 48 \
    --hidden-size 7168 \
    --num-attention-heads 56 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 100 \
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
    --checkpoint-activations \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --no-cuda \
    --use-cpu-initialization \
    --num-workers 0 \
    --no-load-rng \
    $PARALLEL_ARGS \
    $TENSORBOARD_ARGS

