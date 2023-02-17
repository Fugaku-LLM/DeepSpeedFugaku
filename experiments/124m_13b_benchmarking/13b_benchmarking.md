# 13B model benchmarking results

## Overview
### Model hyperparameters
- --num-layers 40 
- --hidden-size 5120 
- --num-attention-heads 40 

### Notations
- MBS = micro batch size
- GBS = global batch size
- Sec/it = seconds per iteration 
- Est. Aggr. PetaFLOPs = TFLOPs * Nodes / 1024

## Experiments

### Sequence Length=1024 \w Activation Checkpointing
| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |Est. Aggr. PetaFLOPs| Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---: | ----: |
|   10 | 13B |1 |  1 |  1 |   10 | 1000 | - MiB | - |  - | - | - |
|   100 | 13B |10 |  1 |  1 |   10 | 1000 | - MiB | - |  - | - | - |
|   1000 | 13B |100 |  1 |  1 |   10 | 1000 | - MiB | - |  - | - | - |
|   2000 | 13B |200 |  1 |  1 |   10 | 2000 | - MiB | - |  - | - | - |
|   2000 (2.2Hz)  | 13B |200 |  1 |  1 |   10 | 2000 | - MiB | - |  - | - | - |

## Comments
