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
| Nodes | Freq   | Size | DP  | MP | PP | MBS  | GBS  |  AC | Mem         | Sec/it | TFLOPs | Est. Aggr. PetaFLOPs| Notes |
| ----: | -----: | ---: | --: | -: | -: | ---: | ---: |  --: | ----------: | -----: | -----: | ------------------: | ----: |
|    48 | 2.0GHz |  30B |  1 |  1 |  48 |   1 | 1536 |  Yes |      - MiB | - |  - | - | - |
|    192 | 2.0GHz |  30B |  4 |  1 |  48 |   1 | 1536 |  Yes |      - MiB | - |  - | - | - |
|    192 | 2.0GHz |  30B |  1 |  4 |  48 |   1 | 1536 |  Yes |      - MiB | - |  - | - | - |
|    192 | 2.0GHz |  30B |  1 |  4 |  48 |   1 | 1536 |  No |      - MiB | - |  - | - | - |
|    192 | 2.0GHz |  30B |  1 |  8 |  24 |   1 | 1536 |  No |      - MiB | - |  - | - | - |
|    192 | 2.0GHz |  30B |  1 |  16 |  12 |   1 | 1536 |  No |      - MiB | - |  - | - | - |
|    1536 | 2.0GHz |  30B |  32 |  1 |  48 |   1 | 1536 | Yes |       - MiB | - |  - | - | - |
|    6144 | 2.0GHz |  30B |  32 |  4 |  48 |   1 | 1536 | No |       - MiB | - |  - | - | - |
