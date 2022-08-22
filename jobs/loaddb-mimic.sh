#!/bin/bash

sbatch <<EOT &
#!/bin/bash

# Request resources --------------
# Graham GPU node: 2 cores, 64G ram
#SBATCH --account def-bentahar
#SBATCH --job-name=$1
#SBATCH --output=console.out
#SBATCH --cpus-per-task=2          # Number of cores (not cpus)sa
#SBATCH --mem=64G               # memory (per node)
#SBATCH --time=0-12:00             # time (DD-HH:MM)

# Setup and run task -------------
module load apptainer/1.0
apptainer exec --env-file variables.env --env WANDB_RUN_NAME=$1 --bind /home/fgrcl/projects/def-bentahar/fgrcl/ML-BP-Estimation/data/mimic-IV:/mnt/mimic4 docker://fgrcl/ml-bp-estimation:$1 python -m mlbpestimation.data.mimic4.initdb
EOT