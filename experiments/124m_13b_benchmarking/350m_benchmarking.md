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

### Sequence Length=1024, PyTorch 1.10
| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |Est. Aggr. PetaFLOPs| Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---: | ----: |
| 4 | 350M | 4 |  1 | 1  |   1 | 512 | - MiB | 371.4 |  0.83 | 0.003 | 03-04 with Activation Checkpointing |
| 8 | 350M | 1 |  8 | 1  |   1 | 512 | - MiB | -  |  - | - | 03-07 |
| 512 | 350M | 512 |  1 | 1  |   1 | 512 | - MiB | 6.7 |  0.47 | 0.23 | 03-07 with Activation Checkpointing|
| 512 | 350M | 512 |  2 | 1  |   1 | 512 | - MiB | - | - | - | 03-07|
| 4096 | 350M | 512 |  8 | 1  |   1 | 512 |4496.4 MiB | 2.1 |  0.14| 0.57 | 03-04 |
