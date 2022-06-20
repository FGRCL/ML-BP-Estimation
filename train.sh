#!/bin/bash

# Request resources --------------
# Graham GPU node: 32 cores, 128G ram, 2 GPUs (12G each)
#SBATCH --account def-bentahar
#SBATCH --gres=gpu:v100l:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=8          # Number of cores (not cpus)
#SBATCH --mem=32G               # memory (per node)
#SBATCH --time=0-24:00             # time (DD-HH:MM)

# Setup and run task -------------
module restore ml
module load gcc/9.3.0 arrow/8.0.0
virtualenv --no-download env
source env/bin/activate
pip install -r requirements.txt

python -m train.py