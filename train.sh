#!/bin/bash
#SBATCH --account def-bentahar
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:00     # DD-HH:MM:SS

module load python/3.8 cuda cudnn

source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

cp -r /home/fgrcl/projects/def-bentahar/fgrcl/ML-BP-Estimation/data/. "$SLURM_TMPDIR"

export DATA_DIRECTORY_PATH=$SLURM_TMPDIR
set -a
source .env
set +a

python -m mlbpestimation.train baseline_window_mimic_preprocessed