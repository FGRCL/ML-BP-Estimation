#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --account def-bentahar
#SBATCH --cpus-per-task=2  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:10     # DD-HH:MM:SS

module load python/3.8 cuda cudnn

source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

export HYDRA_FULL_ERROR=1
echo "$a"
python -m mlbpestimation.train directories=cc wandb=dev "$a"
EOT