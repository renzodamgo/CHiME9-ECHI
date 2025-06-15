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

# Run the prepare, infer and evaluate
echo "Submitting main evaluation job array..."
# Capture the output of the first python command
output=$(python run.py \
    report.run=false \
    evaluate.n_batches=${N_BATCHES} \
    evaluate.batch="range(1,$((${N_BATCHES} + 1)))" \
    hydra/launcher=echi_submitit_slurm \
    --multirun)

echo "$output"

# Extract the job ID. Assumes the output contains a line like "Submitted batch job XXXX"
# or "Submitted job XXXX". Adjust the grep and awk/sed pattern if needed.
JOB_ID=$(echo "$output" | grep -oP 'Submitted (batch )?job \K\d+' || echo "")

if [ -z "$JOB_ID" ]; then
    echo "Failed to retrieve JOB_ID from the first submission. Output was:"
    echo "$output"
    exit 1
fi

echo "Main evaluation job array submitted with ID: ${JOB_ID}"

# Run the report after the previous job has completed successfully
echo "Submitting report job with dependency on job ID: ${JOB_ID}"
python run.py \
    prepare.run=false \
    inference.run=false \
    evaluate.run=false \
    report.run=true \
    hydra.launcher.additional_parameters.dependency=afterok:${JOB_ID}

echo "Report job submitted."
