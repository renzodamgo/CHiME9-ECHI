#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=128:00:00
#SBATCH --output=slurm/logs/inference/%j.out
#SBATCH --mail-user=rwhsutherland1@sheffield.ac.uk
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh

conda activate echi_recipe

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH='$PYTHONPATH:src'

python3 run_enhancement.py device=ha
