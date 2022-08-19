#!/bin/bash

sbatch <<EOT
#!/bin/bash

# Request resources --------------
# Graham GPU node: 32 cores, 128G ram, 2 GPUs (12G each)
#SBATCH --account def-bentahar
#SBATCH --job-name=$1
#SBATCH --output=console.out
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=16          # Number of cores (not cpus)
#SBATCH --mem=128G               # memory (per node)
#SBATCH --time=2-00:00             # time (DD-HH:MM)

# Setup and run task -------------
module load apptainer/1.0 cuda/11.7
apptainer run --nv --env-file variables.env --env WANDB_RUN_NAME=$1 --bind /home/fgrcl/projects/def-bentahar/fgrcl/ML-BP-Estimation/data/mimic-IV:/mnt/mimic4 docker://fgrcl/ml-bp-estimation:$1
EOT