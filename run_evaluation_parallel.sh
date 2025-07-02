#!/bin/bash

set -e
set -u
set -o pipefail

# Function to display usage information
usage() {
    echo "Usage: $0 <EXP_NAME> <ENHANCED_DIR> [OPTIONS] [N_BATCHES]"
    echo "Runs all evaluation stages using a scheduler to parallelize the evaluation stage."
    echo
    echo "Arguments:"
    echo "  EXP_NAME              Name for the experiment (used for output directories)"
    echo "  ENHANCED_DIR          Directory containing enhanced signals"
    echo "  N_BATCHES (optional)  Number of parallel processes to run for evaluation."
    echo "                        Defaults to 40 for slurm, 10 for local."
    echo
    echo "Options:"
    echo "  --launcher TYPE       Launcher type: 'slurm' or 'local'. Defaults to 'slurm'."
    echo "  --dry-run            Show commands that would be run without executing them."
    echo "  -h, --help            Display this help message and exit."
    echo
    echo "Examples:"
    echo "  $0 exp1 /path/to/enhanced                    # SLURM with 40 batches"
    echo "  $0 exp1 /path/to/enhanced --launcher local 8 # Local with 8 batches"
    echo "  $0 exp1 /path/to/enhanced --dry-run          # Show commands without running"
    exit 0
}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Error handling function
handle_error() {
    log "ERROR: Command failed at line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Check for help flag first
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# Check if required arguments are provided
if [[ $# -lt 2 ]]; then
    echo "Error: Missing required arguments."
    echo
    usage
fi

# Get required arguments
EXP_NAME="$1"
ENHANCED_DIR="$2"
shift 2

# Default values
N_BATCHES=""
LAUNCHER="slurm"
DRY_RUN=false

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --launcher)
        LAUNCHER="$2"
        shift 2
        ;;
    --dry-run)
        DRY_RUN=true
        shift
        ;;
    *)
        # Assume it's N_BATCHES if it's a number
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            N_BATCHES="$1"
        else
            echo "Error: Unknown argument '$1'"
            usage
        fi
        shift
        ;;
    esac
done

# Validate launcher type
if [[ "$LAUNCHER" != "slurm" && "$LAUNCHER" != "local" ]]; then
    echo "Error: Launcher must be either 'slurm' or 'local'"
    exit 1
fi

# Set defaults based on launcher if not specified
if [[ -z "$N_BATCHES" ]]; then
    if [[ "$LAUNCHER" == "local" ]]; then
        N_BATCHES=10
    else
        N_BATCHES=40
    fi
fi

# Set the appropriate launcher config
if [[ "$LAUNCHER" == "local" ]]; then
    LAUNCHER_CONFIG="echi_submitit_local"
else
    LAUNCHER_CONFIG="echi_submitit_slurm"
fi

# Function to execute or show commands
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: $*"
    else
        log "Executing: $*"
        "$@"
    fi
}

# Validate that ENHANCED_DIR exists
if [[ ! -d "$ENHANCED_DIR" ]]; then
    log "ERROR: Enhanced directory '$ENHANCED_DIR' does not exist."
    exit 1
fi

log "Configuration:"
log "  Experiment name: $EXP_NAME"
log "  Enhanced directory: $ENHANCED_DIR"
log "  Launcher: $LAUNCHER (config: $LAUNCHER_CONFIG)"
log "  Number of batches: $N_BATCHES"
log "  Dry run: $DRY_RUN"

# Validate Python scripts exist
for script in scripts/evaluation/setup.py scripts/evaluation/validate.py scripts/evaluation/prepare.py scripts/evaluation/evaluate.py scripts/evaluation/report.py; do
    if [[ ! -f "$script" ]]; then
        log "ERROR: Required script not found: $script"
        exit 1
    fi
done

log "Starting evaluation pipeline..."

log "Stage 1: Setup"
run_cmd python scripts/evaluation/setup.py

log "Stage 2: Validation"
run_cmd python scripts/evaluation/validate.py paths.enhancement_output_dir=${ENHANCED_DIR}

log "Stage 3: Preparation"
run_cmd python scripts/evaluation/prepare.py shared.exp_name=${EXP_NAME} paths.enhancement_output_dir=${ENHANCED_DIR}

log "Stage 4: Evaluation (multirun with $N_BATCHES batches)"
run_cmd python scripts/evaluation/evaluate.py \
    paths.enhancement_output_dir=${ENHANCED_DIR} \
    shared.exp_name=${EXP_NAME} \
    evaluate.n_batches=${N_BATCHES} \
    evaluate.batch="range(1,$((${N_BATCHES} + 1)))" \
    hydra/launcher=${LAUNCHER_CONFIG} \
    --multirun

log "Stage 5: Reporting"
run_cmd python scripts/evaluation/report.py shared.exp_name=${EXP_NAME}

log "Evaluation pipeline completed successfully!"
