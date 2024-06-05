#! /bin/bash

eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false

conda activate hqcal_env

python /home/alejandrolc/QuantumSpain/AutoQML/Hierarqcal/qnn.py

conda deactivate