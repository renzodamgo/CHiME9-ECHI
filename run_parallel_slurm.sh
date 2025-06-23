#!/bin/bash

set -e
set -u
set -o pipefail

# Function to display usage information
usage() {
    echo "Usage: $0 [N_BATCHES]"
    echo "Runs all stages using a SLURM scheduler to parallelize the evaluation stage."
    echo
    echo "Arguments:"
    echo "  N_BATCHES (optional): Number of parallel processes to run for evaluation."
    echo "                        Defaults to 40."
    echo
    echo "Options:"
    echo "  -h, --help            Display this help message and exit."
    exit 0
}

# Check for help flag
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# N_BATCHES: Number of parallel processes to run.
# Defaults to 40.
# Set this according to the configuration of your SLURM cluster.
# Usage: ./run_parallel_slurm.sh [N_BATCHES]
N_BATCHES="${1:-40}"

echo "Run setup stage..."
python scripts/setup.py

echo "Run enhance stage..."
python scripts/enhance.py

echo "Run validation stage..."
python scripts/validate.py

echo "Run preparation stage..."
python scripts/prepare.py

echo "Multirun evaluate stage..."
python scripts/evaluate.py \
    evaluate.run=true \
    evaluate.n_batches=${N_BATCHES} \
    evaluate.batch="range(1,$((${N_BATCHES} + 1)))" \
    hydra/launcher=echi_submitit_slurm \
    --multirun

echo "Run reporting stage..."
python scripts/report.py
