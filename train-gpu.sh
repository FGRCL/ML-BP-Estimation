#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --account def-bentahar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:10     # DD-HH:MM:SS

module load python/3.8 cuda cudnn

source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

cp -r /home/fgrcl/projects/def-bentahar/fgrcl/ML-BP-Estimation/data/. "$SLURM_TMPDIR"

python -m mlbpestimation.train directories=cc wandb=dev "$a"
EOT