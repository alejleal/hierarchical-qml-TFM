#! /bin/bash

SLRNAME=slave3

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_DIR=${CONDA_PREFIX}/lib

# export TF_ENABLE_ONEDNN_OPTS=0

# export JAX_ENABLE_X64=True

# export CUDA_VISIBLE_DEVICES=""

python qml_torch.py