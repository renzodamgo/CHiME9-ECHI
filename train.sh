#!/bin/bash
#
#SLUdRM CONFIGURATION FOR A SINGLE, SEQUENTIAL JOB --- 
#
#SBATCH --job-name=train 
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#
# --- LOGGING ---
#
#SBATCH --output=./stanage_logs/train_%A.out
#SBATCH --error=./stanage_logs/train_%A.err
# --- NOTIFICATIONS ---
#SBATCH --mail-user=redamiangomez1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# Load Anaconda module
module load Anaconda3/2024.02-1

# Initialize conda for bash (needed in SLURM environment)
eval "$(conda shell.bash hook)"

# Activate environment
conda activate echi_recipe

# Add Wandb API KEY
export WANDB_API_KEY=51b622235d92829d6361ce45296c77c501810656
export WANDB_PROJECT=echi-train

# Install missing dependencies
echo "Installing dependencies..."
pip install hydra-core

conda install -c conda-forge soxr-python pesq -y

# Set compiler flags for any remaining packages that need compilation
export CFLAGS="-std=c99"

# Install pysepm if not already installed
pip install git+https://github.com/ftshijt/pysepm.git --no-build-isolation --no-deps
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PWD/src:$PYTHONPATH"
# Or alternatively, run from the project directory
cd /mnt/parscratch/users/acp24red/CHiME9-ECHI
echo "Starting enhancement script..."
# Show error outputs on normal output to see all live updates on evaluation_%A.out
exec 2>&1
python run_train.py --config-name main_ha \
  device=ha \
  shared.exp_name=ha-joint \
  dataloading.joint_for=[train] \

echo "Job completed!"
