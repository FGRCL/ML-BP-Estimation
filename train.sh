#!/bin/bash
#SBATCH --account def-bentahar
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64G          # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-03:00     # DD-HH:MM:SS

module load python/3.8 cuda cudnn

poetry export --format=requirements.txt > requirements.txt

rm -rf venv
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index -r requirements.txt

cp /home/fgrcl/projects/def-bentahar/fgrcl/ML-BP-Estimation/data/mimic-IV/physionet.org/files/mimic4wdb/0.1.0/waves "$SLURM_TMPDIR"
export MIMIC_FILE_LOCATION="$SLURM_TMPDIR"

cd $SOURCEDIR
python -m mlbpestimation.train baseline_window_vitaldb