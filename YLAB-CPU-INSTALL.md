Hinadori CPU Installation

## Instal PyTorch 1.10.1 on Hinadori

1. allocate interactive job
    ```bash
    yrun threadripper-3960x_1
    ```
2. module load
    ```bash
    . /etc/profile.d/modules.sh
    module load openmpi/4.1.4-no-cuda
    ```
3. download PyTorch 1.10.1
    ```bash
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch

    git checkout release/1.10

    git submodule sync
    git submodule update --init --recursive --jobs 0
    ```
4. activate virtual environment
    ```bash
    cd ..
    pyenv install 3.9.16  # same version as Fugaku (PyTorch 1.10.1)
    pyenv local 3.9.16

    python -m venv venv
    source venv/bin/activate
    ```
5. pip install
    Please use pip, not conda. (official document says use conda, but it is not working.)
    ```bash
    pip install mpi4py

    pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

    pip install pyyaml
    ```
6. build PyTorch

    clean up
    ```bash
    cd pytorch
    rm -rf build/
    python setup.py clean
    ```

    build
    ```bash
    CMAKE_C_COMPILER=$(which mpicc)
    CMAKE_CXX_COMPILER=$(which mpicxx)

    python setup.py install
    ```

7. check PyTorch version
    ```bash
    module load openmpi/4.1.4-no-cuda
    python -c "import torch; print(torch.__version__)"
    ```

## Build environment for DeepSpeedFugaku on Hinadori

1. allocate interactive job
    ```bash
    yrun threadripper-3960x_1
    ```
2. module load
    ```bash
    . /etc/profile.d/modules.sh
    module load openmpi/4.1.4-no-cuda
    ```
3. activate virtual environment
    ```bash
    source venv/bin/activate
    ```
4. clone DeepSpeed
    ```bash
    rm -rf DeepSpeed # if exists
    git clone git@github.com:microsoft/DeepSpeed.git
    git checkout v0.8.0  # same version as Fugaku deepspeed
    ```

5. install DeepSpeed

    change the whole DeepSpeed/deepspeed/runtime/zero/partition_parameters.py into https://github.com/rioyokotalab/DeepSpeedFugaku/blob/cpu/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py

    and then
    ```bash
    cd DeepSpeed
    pip install pybind11 six regex numpy

    pip install pydantic==1.10.10  # version above 2.o is not working
    pip install pydantic-core==0.42.0 # version above 2.x is not working

    bash install.sh
    ```
