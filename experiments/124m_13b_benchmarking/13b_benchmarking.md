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
| Nodes | Freq   | Size | DP  | TP | PP | MBS  | GBS  | Mem         | Sec/it | TFLOPs | Est. Aggr. PetaFLOPs| Notes |
| ----: | -----: | ---: | --: | -: | -: | ---: | ---: | ----------: | -----: | -----: | ------------------: | ----: |
|    20 | 2.0GHz |  13B |   1 | 20 |  1 |    1 | 1024 | 14272.9 MiB | 7717.7 |   0.72 |              0.0144 |     - |
|   100 | 2.0GHz |  13B |  10 |  1 |  1 |   10 | 1000 |       - MiB | - |  - | - | - |
|  1000 | 2.0GHz |  13B | 100 |  1 |  1 |   10 | 1000 |       - MiB | - |  - | - | - |
|  2000 | 2.0GHz |  13B | 200 |  1 |  1 |   10 | 2000 |       - MiB | - |  - | - | - |
|  2000 | 2.2GHz |  13B | 200 |  1 |  1 |   10 | 2000 |       - MiB | - |  - | - | - |

## Comments
