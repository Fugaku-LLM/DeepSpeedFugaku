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
4. Enable pre-built PyTorch v1.7.0
    ```
	source /vol0004/apps/oss/PyTorch-1.7.0/example/env.src`
    ```
5. Install required Python modules
	```
	export PYTHONUSERBASE=$HOME/work/.local
	export PATH=$PATH:$PYTHONUSERBASE/bin
	pip install --user deepspeed
	pip install --user datasets
    pip install --user nltk
	```
6. Build DeepSpeedFugaku
	```
	git clone git@github.com:idaten459/DeepSpeedFugaku.git
	cd DeepSpeedFugaku/
	python3 setup.py install --user
	```

Prepare training dataset
------------------------

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
    pjsub --interact -L "node=1" -L "rscunit=rscunit_ft01" -L "rscgrp=int" -L "elapse=6:00:00" --sparam "wait-time=600" --mpi "proc=48" -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004`
    ```
4. Download gpt2-vocab.json and gpt2-merges.txt
    ```
    cd dataset
    sh download_vocab.sh
    ```
5. Download dataset and convert to JSON format
    ```
    export HF_DATASETS_CACHE="$HOME/work/DeepSpeedFugaku/.cache"
    mkdir -p $HF_DATASETS_CACHE
    python - << EOF
    from datasets import load_dataset
    train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train')
    train_data.to_json("codeparrot_data.json", lines=True)
    EOF
    ```
6. Preprocess dataset
    ```
    python tools/preprocess_data.py \
           --input dataset/codeparrot_data.json \
           --output-prefix codeparrot \
           --vocab dataset/gpt2-vocab.json \
           --dataset-impl mmap \
           --tokenizer-type GPT2BPETokenizer \
           --merge-file dataset/gpt2-merges.txt \
           --json-keys content \
           --workers 32 \
           --append-eod
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
    newgrp hp190122`
    ```
3. Launch interactive job
    ```
    pjsub --interact -L "node=1" -L "rscunit=rscunit_ft01" -L "rscgrp=int" -L "elapse=6:00:00" --sparam "wait-time=600" --mpi "proc=48" -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
    ```
4. Enable pre-built PyTorch v1.7.0
    ```
	source /vol0004/apps/oss/PyTorch-1.7.0/example/env.src
    ```
5. Export environment variables
    ```
    export PYTHONUSERBASE=$HOME/work/.local
    export PATH=$PATH:$PYTHONUSERBASE/bin
    ```
6. Execute job
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
3. Execute job
    ```
    pjsub run_pretrain_gpt_fugaku.sh
    ```
