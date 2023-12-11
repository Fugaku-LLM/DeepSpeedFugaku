#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM --rsc-list "proc-openfd=65536"
#PJM -L elapse=815:00:00
#PJM -L "node=48x6x48:torus:strict-io"
#PJM -L "freq=2200"
#PJM -L "throttling_state=0"
#PJM -L "issue_state=0"
#PJM -L "ex_pipe_state=0"
#PJM -L "eco_state=0"
#PJM -L "retention_state=0"
#PJM --mpi "proc=1"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp230254
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004:/vol0005:/vol0006
#PJM --llio localtmp-size=30Gi
#PJM --llio sharedtmp-size=50Gi
#PJM -j
#PJM -S

pp=8
tp=3
dp=576
gbs=2304
num_node=13824
hostfile_name="24x2x24x2x3x2_tp${tp}dp${dp}pp${pp}"
param_name="13b_pp${pp}_tp${tp}_dp${dp}_fjpytorch_rankmap_gbs${gbs}_useMPIallreduce"
stdproc_name="jobs/${param_name}/outs/${PJM_JOBID}_n/stdproc"
LP="/local/fcc/inst/other/lib/libtcmalloc.so"

rm /home/u11890/work/rankmap/vcoordfile_${hostfile_name}_fj

llio_transfer /home/u11890/work/rankmap/fjmpi_6d_to_3d.out

mpirun -n ${num_node} /home/u11890/work/rankmap/fjmpi_6d_to_3d.out /home/u11890/work/rankmap/hostfile_${hostfile_name} /home/u11890/work/rankmap/vcoordfile_${hostfile_name}_fj

llio_transfer --purge /home/u11890/work/rankmap/fjmpi_6d_to_3d.out

llio_transfer 13b_pp8_tp3_dp576_inner.sh
llio_transfer /home/u11890/work/1701935794.711074240.fcc.pytorch.y.r1.13_for_a64fx.tar.gz

# execute python code
llio_transfer /home/u11890/work/training/DeepSpeedFugaku/pretrain_gpt.py
/home/system/tool/dir_transfer /home/u11890/work/training/DeepSpeedFugaku/llm-jp-tokenizer
/home/system/tool/dir_transfer /home/u11890/work/training/DeepSpeedFugaku/megatron
/home/system/tool/dir_transfer /home/u11890/work/training/DeepSpeedFugaku/DeepSpeed
/home/system/tool/dir_transfer /home/u11890/work/training/DeepSpeedFugaku/scripts

mpirun -n ${num_node} \
  -mca common_tofu_use_memory_pool 1 \
  -x PATH \
  -x WANDB_INIT_TIMEOUT=3600 \
  -x WANDB__SERVICE_WAIT=3600 \
  -std-proc ${stdproc_name} \
  --vcoordfile /home/u11890/work/rankmap/vcoordfile_${hostfile_name}_fj \
  bash 13b_pp8_tp3_dp576_inner.sh "${LP}"
