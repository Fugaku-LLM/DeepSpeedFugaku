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

### Sequence Length=1024 \w Activation Checkpointing, PyTorch 1.10
| Nodes | Freq   | Size | DP  | TP | PP | MBS  | GBS  | Mem         | Sec/it | TFLOPs | Est. Aggr. PetaFLOPs| Notes |
| ----: | -----: | ---: | --: | -: | -: | ---: | ---: | ----------: | -----: | -----: | ------------------: | ----: |
|    12 | 2.0GHz |  13B |  1 |  1 |  12 |   1 | 1000 |       - MiB | - |  - | - | - |
|    20 | 2.0GHz |  13B |   1 | 20 |  1 |    1 | 1024 | 14272.9 MiB | 7717.7 |   0.72 |              0.0144 |     - |
|    24 | 2.0GHz |  13B |  2 |  1 |  12 |   1 | 1000 |       - MiB | - |  - | - | - |
|    40 | 2.0GHz |  13B |  2 |  20 |  1 |   1 | 1024 |       - MiB | - |  - | - | - |

## Comments
