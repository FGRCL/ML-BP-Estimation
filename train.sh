#!/bin/bash
#SBATCH --account def-bentahar
#SBATCH --gres=gpu:2       # Request GPU "generic resources"
#SBATCH --cpus-per-task=2  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=5-00:00     # DD-HH:MM:SS

module load python/3.8 cuda cudnn

source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

cp -r /home/fgrcl/projects/def-bentahar/fgrcl/ML-BP-Estimation/data/. "$SLURM_TMPDIR"

python -m mlbpestimation.train directories=cc wandb=online hypothesis/model=baseline hypothesis/dataset_loader=mimic_window_saved