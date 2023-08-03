#!/bin/bash

source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin

user_name=$(whoami)
cd /home/$user_name/work/DeepSpeedFugaku

MAX_LENGTH=50
NUM_SAMPLES=35


python evaluation/text_extractor.py \
  --input dataset/wikipedia/merged/ja/ja_merged.json \
  --output evaluation/out/samples.txt \
  --text-max-len $MAX_LENGTH \
  --num-samples $NUM_SAMPLES \
  --random-choice

