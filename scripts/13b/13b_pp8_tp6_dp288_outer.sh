#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=rt"
#PJM --rsc-list "proc-openfd=65536"
#PJM -L elapse=150:00:00
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
#PJM --llio localtmp-size=50Gi
#PJM --llio sio-read-cache=off
#PJM -j
#PJM -S

pp=8
tp=6
dp=288
gbs=2016
num_node=13824
hostfile_name="24x2x24x2x3x2_tp${tp}dp${dp}pp${pp}"
param_name="13b_pp${pp}_tp${tp}_dp${dp}_fjpytorch_rankmap_gbs${gbs}"
stdproc_name="jobs/${param_name}/output.%j/%m/%/1000r/stdproc"
LP="/local/fcc/inst/other/lib/libtcmalloc.so:/vol0503/share/hp230254/allreduce/my_mpi_allreduce_utofu_thresh100m_1214_noprint.so"

#rm /home/u11890/work/rankmap/vcoordfile_${hostfile_name}_fj

#llio_transfer /home/u11890/work/rankmap/fjmpi_6d_to_3d.out

#mpirun -n ${num_node} /home/u11890/work/rankmap/fjmpi_6d_to_3d.out /home/u11890/work/rankmap/hostfile_${hostfile_name} /home/u11890/work/rankmap/vcoordfile_${hostfile_name}_fj

#llio_transfer --purge /home/u11890/work/rankmap/fjmpi_6d_to_3d.out

llio_transfer 13b_pp8_tp6_dp288_inner.sh
llio_transfer /vol0005/mdt3/share/hp230254/pytorch/1703667164.202942381.fcc.pytorch.y.r1.13_for_a64fx.tar.gz
llio_transfer /vol0503/share/hp230254/allreduce/my_mpi_allreduce_utofu_thresh100m_1214_noprint.so

# execute python code
DSF_HOME=/vol0503/data/hp230254/u10270/DeepSpeedFugaku
llio_transfer ${DSF_HOME}/pretrain_gpt.py
/home/system/tool/dir_transfer ${DSF_HOME}/llm-jp-tokenizer
/home/system/tool/dir_transfer ${DSF_HOME}/megatron
/home/system/tool/dir_transfer ${DSF_HOME}/DeepSpeed

#echo "begin llio_transfer dataset idx" `date`
#
## transfer .idx files (12.7 GB)
#DATASET_PATH=/vol0503/data/hp230254/dataset/llm-jp-corpus-v1.0.2/fugaku_13b/binarized/v2_2-code10k_en20k_ja30k
#find $DATASET_PATH -name "*.idx" | xargs -n 1 llio_transfer
#
#echo "end llio_transfer dataset idx" `date`

mpirun -n ${num_node} \
  -mca common_tofu_use_memory_pool 1 \
  -mca coll_base_reduce_commute_safe 1 \
  -x PATH \
  -x WANDB_INIT_TIMEOUT=3600 \
  -x WANDB__SERVICE_WAIT=3600 \
  -std-proc ${stdproc_name} \
  --vcoordfile /vol0503/share/hp230254/rankmap/vcoordfile_${hostfile_name}_fj \
  bash 13b_pp8_tp6_dp288_inner.sh "${LP}"

