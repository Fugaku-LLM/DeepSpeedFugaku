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
|    48 | 2.0GHz |  30B |  1 |  1 |  48 |   1 | 1536 |  Yes |      - MiB | 17212.8 |  0.95 | 0.04 | Stopped after the first iteration.   4/22 |
|    192 | 2.0GHz |  30B |  1 |  4 |  48 |   1 | 1536 |  Yes |      - MiB | 3720.0 |   1.10 | 0.2 |  4/23 |
|    384 | 2.0GHz |  30B |  1 |  8 |  48 |   1 | 1536 |  Yes |      - MiB | - |   - | - | 4/23 |
|    384 | 2.0GHz |  30B |  1 |  8 |  48 |   1 | 1536 |  No |      - MiB | - |   - | - | 4/23 |
|    3072 | 2.0GHz |  30B |  16 |  4 |  48 |   1 | 1536 |  Yes |      16632.9 MiB | 313.6 |   0.82 | 2.5 | 4/23 |
