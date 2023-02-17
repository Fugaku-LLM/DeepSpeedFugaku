# Fugaku Config Sweep

### 124M model, Sequence Length=1024
| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |Est. Aggr. PetaFLOPs| Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---: | ----: |
|   1 | 124M |1 |  1 |  1 |   1 | 1024 |  7718.0 MiB | 1489.7 | 0.56| 0.0005 |02-15 |
|   1 (2.2Hz) | 124M |1 |  1 |  1 |   1 | 1024| 7701.2 MiB | 1261.2 | 0.66| 0.0006 |02-15 |
|   1 (LD_PRELOAD=libtcmalloc.so) | 124M |1 |  1 |  1 |   1 | 1024 |  - | - | -| - | - |
|   1 ("retention_state=0")| 124M |1 |  1 |  1 |   1 | 1024 |  - | - | -| - | - |
|   1 (max-proc-per-node=4)| 124M |4 |  1 |  1 |   1 | 1024 |  - | - | -| - | - |
|   1 (max-proc-per-node=4)| 124M |1 |  4 |  1 |   1 | 1024 |  - | - | -| - | - |
