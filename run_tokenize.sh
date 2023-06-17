#!/bin/bash -x
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=72:00:00
#PJM -L "node=1"
#PJM -g hp190122
#PJM --mpi "max-proc-per-node=4"
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM --name tokenize
#PJM -j
#PJM -S

# activate pytorch environment
source /home/u11887/work/PyTorch-1.10.1/env.src

export OUTDIR=data/wikipedia/binarized/gpt-2/second
mkdir -p $OUTDIR

# Tokenize and binarize Japanese
python tools/preprocess_data.py \
  --input data/wikipedia/merged/ja/ja_merged.json \
  --output-prefix $OUTDIR/ja_wiki \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --dataset-impl mmap \
  --tokenizer-type GPT2BPETokenizer \
  --workers 16 \
  --append-eod
