#!/bin/bash
#YBATCH -r a100_1
#SBATCH --job-name=tokenize
#SBATCH --time=2-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

JA_VOCAB_SIZE=40
EN_VOCAB_SIZE=10

source .env/bin/activate

# Set the output directory:
export OUTDIR=datasets/wikipedia/binarized/v1_ja${JA_VOCAB_SIZE}K_en${EN_VOCAB_SIZE}K
mkdir -p $OUTDIR
export MODELDIR=tokenizer/models/cc100ja1GB_cc100en1GB/cc100_ja${JA_VOCAB_SIZE}K_en${EN_VOCAB_SIZE}K.symbolRemoved.vocab.reestimated.model

# Tokenize and binarize Japanese
python tools/preprocess_data.py \
  --input dataset/wikipedia/merged/ja/ja_merged.json \
  --output-prefix $OUTDIR/ja_wiki \
  --vocab-file $MODELDIR \
  --dataset-impl mmap \
  --tokenizer-type JapaneseSentencePiece \
  --workers 64 \
  --append-eod

# Tokenize and binarize English
python tools/preprocess_data.py \
  --input dataset/wikipedia/merged/en/en_merged.json \
  --output-prefix $OUTDIR/en_wiki \
  --vocab-file $MODELDIR \
  --dataset-impl mmap \
  --tokenizer-type JapaneseSentencePiece \
  --workers 64 \
  --append-eod
