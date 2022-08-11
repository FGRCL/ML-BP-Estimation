#!/bin/bash

sbatch <<EOT
#!/bin/bash

# Request resources --------------
# Graham GPU node: 32 cores, 128G ram, 2 GPUs (12G each)
#SBATCH --account def-bentahar
#SBATCH --job-name=$1
#SBATCH --output=console.out
#SBATCH --wait
#SBATCH --gres=gpu:v100l:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=8          # Number of cores (not cpus)sa
#SBATCH --mem=32G               # memory (per node)
#SBATCH --time=0-05:00             # time (DD-HH:MM)

# Setup and run task -------------
module load scipy-stack/2022a python/3.10 python-build-bundle/2022a
poetry export -f requirements.txt --output requirements.txt
virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt
python -m mlbpestimation.data.mimic4.initdb
EOT