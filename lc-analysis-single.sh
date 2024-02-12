#!/bin/bash

sbatch <<EOT
#!/bin/sh
#SBATCH --account def-bentahar
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=125G      # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=3-00:00     # DD-HH:MM:SS
#SBATCH --output=/home/fgrcl/scratch/job-logs/slurm-%A_%a.out

module load python/3.10 cuda cudnn

source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

export $(cat .env | xargs)
python -m mlbpestimation.train \
  directories=cc \
  wandb=online \
  hypothesis/dataset/source=tfrecord \
  hypothesis.dataset.source.dataset_name=heartbeat-records \
  hypothesis/dataset/decorators=[subsampletrain] \
  hypothesis.dataset.decorators.subsampletrain.training_preprocessing.sample_rate=$1 \
  hypothesis/model=mlp \
  wandb=online \
  wandb.group=lc-analysis \
  hypothesis/optimization=long
EOT
