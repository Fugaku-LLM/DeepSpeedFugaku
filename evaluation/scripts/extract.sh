#!/bin/bash

MAX_LENGTH=50
NUM_SAMPLES=35

source .env/bin/activate

python evaluation/text_extractor.py \
  --input dataset/wikipedia/merged/ja/ja_merged.json \
  --output evaluation/out/samples.txt \
  --text-max-len $MAX_LENGTH \
  --num-samples $NUM_SAMPLES \
  --random-choice

