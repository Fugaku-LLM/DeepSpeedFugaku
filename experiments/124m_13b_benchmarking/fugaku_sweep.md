# Fugaku Config Sweep

### 124M model, Sequence Length=1024
| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |Est. Aggr. PetaFLOPs| Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---: | ----: |
|   1 | 124M |1 |  1 |  1 |   1 | 1024 |  7718.0 MiB | 1489.7 | 0.56| 0.0005 |02-15 |
|   1 ("retention_state=0")| 124M |1 |  1 |  1 |   1 | 1024 | - MiB  | 1401.1 | 0.60|  0.0006 | 02-15 |
|   1 (2.2Hz) | 124M |1 |  1 |  1 |   1 | 1024| 7701.2 MiB | 1261.2 | 0.66| 0.0006 |02-15 |
|   1 (LD_PRELOAD=libtcmalloc.so) | 124M |1 |  1 |  1 |   1 | 1024 | - MiB | 976.2 | 0.86 | 0.0008 | 02-15 |
|   1 (2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")| 124M |1 |  1 |  1 |   1 | 1024 |  - | 827.2 | 1.01| 0.001 | 02-15 |
|   1 (max-proc-per-node=4)| 124M |4 |  1 |  1 |   1 | 1024 |  22266.2 MiB | 721.7 | 1.16^| 0.001 | 02-15 |
|   1 (max-proc-per-node=4)| 124M |1 |  4 |  1 |   1 | 1024 |  - | - | -| - | 02-15 |
|   1 (max-proc-per-node=4, 2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")| 124M |4 |  1 |  1 |   1 | 1024 | - |522.7 | 1.60^ | 0.001| 02-15 |
|   1 (max-proc-per-node=4, 2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")| 124M |1 |  4 |  1 |   1 | 1024 |  - | - | -| - | 02-15 |

### 1.3B model, Sequence Length=1024, \w Activation Checkpointing
| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |Est. Aggr. PetaFLOPs| Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---: | ----: |
|   1 | 1.3B |1 |  1 |  1 |   1 | 1024 |28406.7 MiB | 11727.9 |  0.99 | 0.001 | 02-14 |
|   1 ("retention_state=0") | 1.3B | 1|  1 |  1 |   1 | 1024 |29081.7 MiB |  10023.9 | 1.13 | 0.001| 02-15 |
|   1 (2.2Hz) | 1.3B | 1|  1 |  1 |   1 | 1024 |28356.1 MiB | 10532.4 |  1.10 |  0.001 | 02-15 |
|   1 (LD_PRELOAD=libtcmalloc.so)^^  | 1.3B | 1|  1 |  1 |   1 | 1024 |- MiB |  - | - | -| - | 02-15 |
|   1 (2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")^^  | 1.3B | 1|  1 |  1 |   1 | 1024 |- MiB |  - | - | -| - | 02-15 |
|   1 (max-proc-per-node=4)^^ |1.3B | 1|  4 |  1 |   1 | 1024 |- MiB |  - | - | -| - | 02-15 |
|   1 (max-proc-per-node=4, 2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")^^ |1.3B | 1|  4 |  1 |   1 | 1024 |- MiB |  - | - | -| - | 02-15 |

### Comments
- ^ Estimated from the per-core-group performance
- ^^ Stuck after the first epoch
- max-proc-per-node=4 with mp=4 does not work.

