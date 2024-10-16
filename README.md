# Environment Setup
1. Create conda env
```
conda create -n lens
conda activate lens
conda install python==3.8.13
```

2. Install pytorch and torch scatter
Be aware that pytorch version should be carfully selected based on YOUR environment.
Following command shows our environment.
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
```

3. Install Dependencies
```
pip install -r requirements
```

4. make cuda available (Our code is tested with CUDA 11.4)
```
export CUDA_HOME=/apps/cuda/11.4
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export CUDNN_LIB_DIR=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include/
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```


5. link dataset
```
cd LensNeRF
mkdir data
cd data
git lfs install
git clone https://huggingface.co/datasets/emjay73/lensnerf_dataset

# Some noticeable Errors 
1)Index error 
```
in _get_cuda_arch_flags
    arch_list[-1] += '+PTX'
IndexError: list index out of range
```

following might help
```
# https://github.com/pytorch/extension-cpp/issues/71
export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
```

# Thanks Note
Our code is based on the DirectVoxGO implementation.
Huge thanks for the amaing work by the authors of DVGO!

Direct Voxel Grid Optimization (CVPR2022 Oral, [project page](https://sunset1995.github.io/dvgo/), [DVGO paper](https://arxiv.org/abs/2111.11215), [DVGO v2 paper](https://arxiv.org/abs/2206.05085)).

