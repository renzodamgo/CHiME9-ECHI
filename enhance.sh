#!/bin/bash
#
#SLUdRM CONFIGURATION FOR A SINGLE, SEQUENTIAL JOB --- 
#
#SBATCH --job-name=enhance 
#
# --- ADD THIS LINE TO SELECT THE GPU PARTITION ---
#SBATCH --partition=gpu
# --- REQUEST A GPU ---
#SBATCH --gres=gpu:1
#
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#
# --- LOGGING ---
#
#SBATCH --output=./stanage_logs/enhance_%A.out
#SBATCH --error=./stanage_logs/enhance_%A.err
# --- NOTIFICATIONS ---
#SBATCH --mail-user=redamiangomez1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# Load Anaconda module
module load Anaconda3/2024.02-1

# Initialize conda for bash (needed in SLURM environment)
eval "$(conda shell.bash hook)"

# Activate environment
conda activate echi_recipe

# Install missing dependencies
echo "Installing dependencies..."
pip install hydra-core

conda install -c conda-forge soxr-python pesq -y

# Set compiler flags for any remaining packages that need compilation
export CFLAGS="-std=c99"

# Install pysepm if not already installed
pip install git+https://github.com/ftshijt/pysepm.git --no-build-isolation --no-deps
export PYTHONPATH="$PWD/src:$PYTHONPATH"
# Or alternatively, run from the project directory
cd /mnt/parscratch/users/acp24red/CHiME9-ECHI
echo "Starting enhancement script..."
python run_enhancement.py device=ha

echo "Job completed!"
