#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --account def-bentahar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=1-00:00     # DD-HH:MM:SS
#SBATCH --output=/home/fgrcl/scratch/job-logs/slurm-%A_%a.out

module load python/3.10 cuda cudnn

source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

export $(cat .env | xargs)
python -m mlbpestimation.train directories=cc wandb=online $@
EOT