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

echo "Run prepare stage..."
python run.py \
    prepare.run=true \
    enhance.run=false \
    evaluate.run=false \
    report.run=false

echo "Run enhance stage..."
python run.py \
    prepare.run=false \
    enhance.run=true \
    evaluate.run=false \
    report.run=false

echo "Multirun evaluate stage..."
python run.py \
    prepare.run=false \
    enhance.run=false \
    evaluate.run=true \
    report.run=false \
    evaluate.n_batches=${N_BATCHES} \
    evaluate.batch="range(1,$((${N_BATCHES} + 1)))" \
    hydra/launcher=echi_submitit_slurm \
    --multirun

echo "Run reporting stage..."
python run.py \
    prepare.run=false \
    enhance.run=false \
    evaluate.run=false \
    report.run=true

echo "$output"
