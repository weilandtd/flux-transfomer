#!/bin/bash
# Submit the full SigmaFlux training workflow as a SLURM pipeline.
#
# Usage (from repo root):
#   bash paper/slurm_submit.sh --output-dir /path/to/output
#   bash paper/slurm_submit.sh --output-dir /path/to/output --force   # re-run even if outputs exist
#
# Note: output_dir should be on scratch space for performance
#
# Jobs are chained with dependencies:
#   Job 1 (sample)  →  Job 2 (generate, array 20×5000)  →  Job 3 (train GPU)

set -e

# Parse command-line arguments
OUTPUT_DIR=""
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash paper/slurm_submit.sh --output-dir /path/to/output [--force]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir is required"
    echo "Usage: bash paper/slurm_submit.sh --output-dir /path/to/output [--force]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper"
CONFIG="$PAPER_DIR/io_config.yaml"

# ── Module and conda environment configuration ──────────────────────────────
# Set conda environment name (change if using a different environment)
CONDA_ENV="sigmaflux"

# Load required HPC modules
module load anaconda3/2025.12
module load cplex/22.1.0

# Initialize conda (adjust to your conda installation)
if [ -z "$CONDA_PREFIX" ]; then
    eval "$(conda shell.bash hook)"
fi

# Read output_model from config
OUTPUT_MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('output_model','transformer.pt'))")

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

DATESTAMP=$(date +%Y%m%d)

echo "=========================================="
echo "SigmaFlux Training Pipeline"
echo "=========================================="
echo "Output root  : $OUTPUT_DIR"
echo "Force regen  : $FORCE"
echo "=========================================="
echo ""

# ── Job 1: Sample flux distributions ─────────────────────────────────────────
JOB1_ID=$(sbatch --parsable \
    --job-name=sf_sample \
    --partition=cpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=08:00:00 \
    --output="$OUTPUT_DIR/logs/sample_%j.out" \
    --error="$OUTPUT_DIR/logs/sample_%j.err" \
    --wrap="
module load anaconda3/2025.12
module load cplex/22.1.0
RUN_DIR=$OUTPUT_DIR/${DATESTAMP}_\$SLURM_JOB_ID
mkdir -p \$RUN_DIR/logs
eval \"\$(conda shell.bash hook)\"
export PYTHONPATH=$REPO_ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}
conda run -n $CONDA_ENV python $PAPER_DIR/sample_flux.py --config $CONFIG --output-dir \$RUN_DIR")
echo "[Job 1] Submitted flux sampling: job $JOB1_ID"

RUN_TAG="${DATESTAMP}_${JOB1_ID}"
RUN_DIR="$OUTPUT_DIR/$RUN_TAG"
FLUX_CSV="$RUN_DIR/flux_samples.csv"

echo "Run folder   : $RUN_DIR"
echo "Output model : $RUN_DIR/$OUTPUT_MODEL"

# ── Job 2: Generate labeling data (array) ────────────────────────────────────
N_SAMPLES=100000
CHUNK=5000
N_TASKS=$((N_SAMPLES / CHUNK))   # = 20

HDF5_CHECK="$RUN_DIR/wb_model_labeling_data_0_${CHUNK}.h5"
DEPENDENCY_2=""
[[ -n "$JOB1_ID" ]] && DEPENDENCY_2="--dependency=afterok:$JOB1_ID"

if [[ "$FORCE" == false && -f "$HDF5_CHECK" ]]; then
    echo "[Job 2] HDF5 files already exist, skipping."
    JOB2_ID=""
else
    JOB2_ID=$(sbatch --parsable \
        --job-name=sf_generate \
        --partition=cpu \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=16G \
        --time=01:00:00 \
        --array="0-$((N_TASKS - 1))" \
        --output="$RUN_DIR/logs/generate_%A_%a.out" \
        --error="$RUN_DIR/logs/generate_%A_%a.err" \
        $DEPENDENCY_2 \
        --wrap="
    module load anaconda3/2025.12
    module load cplex/22.1.0
mkdir -p $RUN_DIR/logs
eval \"\$(conda shell.bash hook)\"
export PYTHONPATH=$REPO_ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}
START_IDX=\$((SLURM_ARRAY_TASK_ID * $CHUNK))
END_IDX=\$((START_IDX + $CHUNK))
conda run -n $CONDA_ENV python $PAPER_DIR/generate_training_data.py \
    --config $CONFIG \
    --output-dir $RUN_DIR \
    --start-idx \$START_IDX \
    --end-idx \$END_IDX")
    echo "[Job 2] Submitted labeling generation array (${N_TASKS} tasks × ${CHUNK} samples): job $JOB2_ID"
fi

# ── Job 3: Train transformer on GPU ──────────────────────────────────────────
DEPENDENCY_3=""
[[ -n "$JOB2_ID" ]] && DEPENDENCY_3="--dependency=afterok:$JOB2_ID"

JOB3_ID=$(sbatch --parsable \
    --job-name=sf_train \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --mem=32G \
    --time=00:59:00 \
    --constraint=gpu80\
    --output="$RUN_DIR/logs/train_%j.out" \
    --error="$RUN_DIR/logs/train_%j.err" \
    $DEPENDENCY_3 \
    --wrap="
module load anaconda3/2025.12
module load cplex/22.1.0
mkdir -p $RUN_DIR/logs
eval \"\$(conda shell.bash hook)\"
export PYTHONPATH=$REPO_ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}
conda run -n $CONDA_ENV python $PAPER_DIR/train_ai_model.py --config $CONFIG --data-dir $RUN_DIR")
echo "[Job 3] Submitted training: job $JOB3_ID"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $RUN_DIR/logs/train_*.out"
