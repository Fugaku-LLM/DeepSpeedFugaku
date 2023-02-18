# Fugaku Config Sweep

### 124M model, Sequence Length=1024, PyTorch 1.10
| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |Est. Aggr. PetaFLOPs| Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---: | ----: |
|   1 (max-proc-per-node=4)| 124M |4 |  1 |  1 |   1 | 1024 | 28827.7 MiB |613.2 | 1.36^/^^| 0.001| 02-15, 21103755 |
|   1 (max-proc-per-node=4)| 124M |1 |  4 |  1 |   1 | 1024 |  - | - | -| - | 02-15, 21104749 |
|   1 (max-proc-per-node=4, 2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")| 124M |4 |  1 |  1 |   1 | 1024 |25201.0 MiB |  363.6 | 2.28^ | 0.002| 02-15 |
| 256 (max-proc-per-node=4) | 124M | 1024 |  1 | 1  |   1 | 1024 | -  | - | -| -| 02-15 |
| 512 | 124M | 512 |  1 | 1  |   1 | 1024 |8223.2 MiB | 3.7 | 0.43 | 0.215 |02-15 |
| 512 (max-proc-per-node=4)| 124M | 512 |  4 |  1 |   1 | 1024 |  - | - | -| - | - |
| 512 (2.2Hz) | 124M | 512 |  1 | 1  |   1 | 1024| 8347.0 MiB | 3.2| 0.50| - |02-15 |
| 512 (LD_PRELOAD=libtcmalloc.so) | 124M | 512 |  1 | 1  |   1 | 1024| - MiB | -| -| - |02-15, 21103942|
| 512 (2.2Hz,  "retention_state=0") | 124M | 512 |  1 | 1  |   1 | 1024| 8362.1 MiB | 3.2|  0.50| - |02-15 |
| 512  (2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")|124M | 512 |  1 | 1  |   2 | 1024| - MiB |  14.39†| 0.11| - |02-15|


### 124M model, Sequence Length=1024, PyTorch 1.7
| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |Est. Aggr. PetaFLOPs| Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---: | ----: |
|   1 | 124M |1 |  1 |  1 |   1 | 1024 |  7718.0 MiB | 1489.7 | 0.56| 0.0005 |02-15 |
|   1 ("retention_state=0")| 124M |1 |  1 |  1 |   1 | 1024 | - MiB  | 1401.1 | 0.60|  0.0006 | 02-15 |
|   1 (2.2Hz) | 124M |1 |  1 |  1 |   1 | 1024| 7701.2 MiB | 1261.2 | 0.66| 0.0006 |02-15 |
|   1 (LD_PRELOAD=libtcmalloc.so) | 124M |1 |  1 |  1 |   1 | 1024 | - MiB | 976.2 | 0.86 | 0.0008 | 02-15 |
|   1 (2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")| 124M |1 |  1 |  1 |   1 | 1024 |  - | 827.2 | 1.01| 0.001 | 02-15 |
|   1 (max-proc-per-node=4)| 124M |4 |  1 |  1 |   1 | 1024 |  22266.2 MiB | 721.7 | 1.16^| 0.001 | 02-15 |
|   1 (max-proc-per-node=4, 2.2Hz, LD_PRELOAD=libtcmalloc.so, "retention_state=0")| 124M |4 |  1 |  1 |   1 | 1024 | - |522.7 | 1.60^ | 0.001| 02-15 |


### 1.3B model, Sequence Length=1024, \w Activation Checkpointing, PyTorch 1.7
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
- † Forward pass and backward reduce are taking a lot of time.
<img width="618" alt="Screen Shot 2023-02-18 at 15 35 50" src="https://user-images.githubusercontent.com/18011504/219896545-a059496d-00af-44be-b3dd-52734865400e.png">

- max-proc-per-node=4 with mp=4 does not work.

