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

# Copyright (c) 2023, Tokyo Institute of Technology.  All rights reserved.
# Copyright (c) 2023, Riken.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
