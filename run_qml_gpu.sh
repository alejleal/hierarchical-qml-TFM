#! /bin/bash

SLRNAME=slave2

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_DIR=${CONDA_PREFIX}/lib

# export TF_ENABLE_ONEDNN_OPTS=0
echo "GPU"
python qml_torch.py "lightning.gpu"

export CUDA_VISIBLE_DEVICES=""

echo "no GPU"
python qml_torch.py "default.qubit.torch"