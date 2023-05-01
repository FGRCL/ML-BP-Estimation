#!/bin/bash
#SBATCH --account def-bentahar
#SBATCH --gres=gpu:2       # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=4-00:00     # DD-HH:MM:SS

module load python/3.8 cuda cudnn

source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

cp -r /home/fgrcl/projects/def-bentahar/fgrcl/ML-BP-Estimation/data/mimic-IV/physionet.org/files/mimic4wdb/0.1.0/waves "$SLURM_TMPDIR"

export MIMIC_FILE_LOCATION=$SLURM_TMPDIR
set -a
source .env
set +a

python -m mlbpestimation.train resnet_window_mimic_preprocessed