#!/bin/bash

set -e
set -u
set -o pipefail

# Function to display usage information
usage() {
    echo "Usage: $0 [N_BATCHES]"
    echo "Runs the prepare, enhance, and evaluate stages in parallel, followed by the report stage."
    echo
    echo "Arguments:"
    echo "  N_BATCHES (optional): Number of parallel processes to run for evaluation."
    echo "                        Defaults to 10."
    echo "                        Set this according to the number of CPU cores available."
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
# Defaults to 10.
# Set this according to the number of CPU cores available on your local machine.
# Usage: ./run_parallel_local.sh [N_BATCHES]
N_BATCHES="${1:-10}"

# Run the prepare, enhance and evaluate
python run.py \
    report.run=false \
    evaluate.n_batches=${N_BATCHES} \
    evaluate.batch="range(1,$((${N_BATCHES} + 1)))" \
    hydra/launcher=echi_submitit_local \
    --multirun

# Run the report after the previous job has completed successfully
python run.py \
    prepare.run=false \
    enhance.run=false \
    evaluate.run=false \
    report.run=true
