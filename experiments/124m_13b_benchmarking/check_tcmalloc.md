# Check for the performance of w/ or w/o tcmalloc

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

### 124M model, Sequence Length=1024 \w Activation Checkpointing, PyTorch 1.10, \w LD_PRELOAD=libtcmalloc.so
| Nodes | Freq   | Size | DP   | TP   | PP   | MBS  | GBS  | Mem         | Sec/it | TFLOPs | Est. Aggr. PetaFLOPs| Notes |
| ----: | -----: | ---: | ---: | ---: | ---: | ---: | ---: | ----------: | -----: | -----: | ------------------: | ----: |
|     1 | 2.0GHz | 124M |    1 |    1 |    1 |    1 | 1024 |  3769.4 MiB | 1020.3 |   1.09 |              0.0010 |     - |
|     2 | 2.0GHz | 124M |    2 |    1 |    1 |    1 | 1024 |  3800.9 MiB |  517.7 |   1.08 |              0.0021 |     - |
|     4 | 2.0GHz | 124M |    4 |    1 |    1 |    1 | 1024 |  4089.5 MiB |  259.9 |   1.07 |              0.0041 |     - |
|     8 | 2.0GHz | 124M |    8 |    1 |    1 |    1 | 1024 |  3925.4 MiB |  130.9 |   1.06 |              0.0082 |     - |
|    16 | 2.0GHz | 124M |   16 |    1 |    1 |    1 | 1024 |  3947.0 MiB |   66.9 |   1.04 |              0.0162 |     - |
|    32 | 2.0GHz | 124M |   32 |    1 |    1 |    1 | 1024 |  4039.7 MiB |   35.8 |   0.97 |              0.0303 |     - |
|    64 | 2.0GHz | 124M |   64 |    1 |    1 |    1 | 1024 |  4110.1 MiB |   17.6 |   0.98 |              0.0612 |     - |

### 124M model, Sequence Length=1024 \w Activation Checkpointing, PyTorch 1.10, w\o LD_PRELOAD=libtcmalloc.so
| Nodes | Freq   | Size | DP   | TP   | PP   | MBS  | GBS  | Mem         | Sec/it | TFLOPs | Est. Aggr. PetaFLOPs| Notes |
| ----: | -----: | ---: | ---: | ---: | ---: | ---: | ---: | ----------: | -----: | -----: | ------------------: | ----: |
|     1 | 2.0GHz | 124M |    1 |    1 |    1 |    1 | 1024 |  4504.5 MiB | 1321.5 |   0.84 |              0.0008 |     - |
|     2 | 2.0GHz | 124M |    2 |    1 |    1 |    1 | 1024 |  4470.4 MiB |  832.0 |   0.67 |              0.0013 |     - |
|     4 | 2.0GHz | 124M |    4 |    1 |    1 |    1 | 1024 |  4506.0 MiB |  448.6 |   0.62 |              0.0024 |     - |
|     8 | 2.0GHz | 124M |    8 |    1 |    1 |    1 | 1024 |  4511.7 MiB |  217.5 |   0.64 |              0.0050 |     - |
|    16 | 2.0GHz | 124M |   16 |    1 |    1 |    1 | 1024 |  4510.0 MiB |  113.1 |   0.62 |              0.0096 |     - |
|    32 | 2.0GHz | 124M |   32 |    1 |    1 |    1 | 1024 |  4566.6 MiB |   60.2 |   0.58 |              0.0181 |     - |
|    64 | 2.0GHz | 124M |   64 |    1 |    1 |    1 | 1024 |  4497.7 MiB |   31.2 |   0.56 |              0.0350 |     - |

## Comments
