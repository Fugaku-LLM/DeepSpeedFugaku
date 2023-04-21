# 30B model benchmarking results

## Overview
### Model hyperparameters
- --num-layers 48 
- --hidden-size 7168 
- --num-attention-heads 56 

### Notations
- MBS = micro batch size
- GBS = global batch size
- Sec/it = seconds per iteration 
- Est. Aggr. PetaFLOPs = TFLOPs * Nodes / 1024

## Experiments

### Sequence Length=2048 \w Activation Checkpointing, PyTorch 1.10
| Nodes | Freq   | Size | DP  | TP | PP | MBS  | GBS  | Mem         | Sec/it | TFLOPs | Est. Aggr. PetaFLOPs| Notes |
| ----: | -----: | ---: | --: | -: | -: | ---: | ---: | ----------: | -----: | -----: | ------------------: | ----: |
|    32 | 2.0GHz |  13B |  1 |  1 |  12 |   1 | 1000 |       - MiB | - |  - | - | - |
