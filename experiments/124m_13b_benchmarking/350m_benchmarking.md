# 350M model benchmarking results

## Overview
### Model hyperparameters
- --num-layers 12 
- --hidden-size 768 
- --num-attention-heads 12 

### Notations
- MBS = micro batch size
- GBS = global batch size
- Sec/it = seconds per iteration 
- Est. Aggr. PetaFLOPs = TFLOPs * Nodes / 1024

## Experiments

### Sequence Length=1024, w\o Activation Checkpointing, PyTorch 1.10
| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |Est. Aggr. PetaFLOPs| Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---: | ----: |
| 4 | 124M | 4 |  8 | 1  |   1 | 512 | - MiB | -  |  - | - |- |
| 4096 | 124M | 512 |  8 | 1  |   1 | 512 | - MiB | 2.1 |  0.14| - |- |
