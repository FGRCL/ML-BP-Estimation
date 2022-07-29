#!/bin/bash

# Request resources --------------
# Graham GPU node: 32 cores, 128G ram, 2 GPUs (12G each)
#SBATCH --account def-bentahar
#SBATCH --gres=gpu:v100l:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=8          # Number of cores (not cpus)
#SBATCH --mem=64G               # memory (per node)
#SBATCH --time=0-02:00             # time (DD-HH:MM)

# Setup and run task -------------
module load apptainer/1.0
apptainer run --nv docker://fgrcl/ml-bp-estimation:"$IMAGE_TAG"
