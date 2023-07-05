Installation on Fugaku
======================

Install DeepSpeedFugaku and Python modules
------------------------------------------

0. Make work directory (If you have never done)
    ```
    WORK="/data/hp190122/users/$(id -u -n)/work"
    mkdir -p $WORK
    ln -s $WORK $HOME/work
    ```
1. Move to work directory
    ```
    cd $HOME/work
    ```
2. Change user group
    ```
    newgrp hp190122
    ```
3. Launch interactive job
    ```
    pjsub --interact -L "node=1" -L "rscunit=rscunit_ft01" -L "rscgrp=int" -L "elapse=6:00:00" --sparam "wait-time=600" --mpi "proc=48" -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
    ```
4. Enable pre-built PyTorch v1.10.1
    ```
	  source /data/hp190122/share/PyTorch-1.10.1/env.src
    ```
5. Install required Python modules
    ```
    export PYTHONUSERBASE=$HOME/work/.local
    export PATH=$PATH:$PYTHONUSERBASE/bin
    pip3 install --user deepspeed
    pip3 install --user datasets
    pip3 install --user nltk
    ```
6. Build DeepSpeedFugaku
    ```
    git clone git@github.com:rioyokotalab/DeepSpeedFugaku.git
    cd DeepSpeedFugaku/

    git switch training/feature/benchmark-ja-wiki
    git branch <user-sub-team-name>/feature/<branch-name>
    git switch <user-sub-team-name>/feature/<branch-name>

    python3 setup.py install --user
    ```

Execute pre-training
--------------------

### Interactive job

1. Move to DeepSpeedFugaku directory
    ```
    cd $HOME/work/DeepSpeedFugaku
    ```
2. Change user group
    ```
    newgrp hp190122
    ```
3. Launch interactive job
    ```
    pjsub --interact -L "node=1" -L "rscunit=rscunit_ft01" -L "rscgrp=int" -L "elapse=6:00:00" --sparam "wait-time=600" --mpi "proc=48" -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
    ```
4. Enable pre-built PyTorch v1.10.1
    ```
	source /data/hp190122/share/PyTorch-1.10.1/env.src
    ```
5. Export environment variables
    ```
    export PYTHONUSERBASE=$HOME/work/.local
    export PATH=$PATH:$PYTHONUSERBASE/bin
6. download vocab file and merge file
    ```
    cd dataset
    bash download_vocab.sh
    cd ..
    ```
7. make megatron/data/helpers.cpp
    ```
    cd megatron/data
    make
    cd ../..
    ```
8. Execute job
    ```
    sh run_pretrain_gpt_fugaku.sh
    ```

### Batch job

1. Move to DeepSpeedFugaku directory
    ```
    cd $HOME/work/DeepSpeedFugaku
    ```
2. Change user group
    ```
    newgrp hp190122
    ```
3. Submit job
    ```
    pjsub run_pretrain_gpt_fugaku.sh
    ```
