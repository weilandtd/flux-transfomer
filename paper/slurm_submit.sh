#!/bin/bash
# Submit the full SigmaFlux training workflow as a SLURM pipeline.
#
# Usage (from repo root):
#   bash paper/slurm_submit.sh
#   bash paper/slurm_submit.sh --force   # re-run even if outputs exist
#
# Jobs are chained with dependencies:
#   Job 1 (sample)  →  Job 2 (generate, array 20×5000)  →  Job 3 (train GPU)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper"
CONFIG="$PAPER_DIR/io_config.yaml"
PYTHON="/scratch/gpfs/MONA/mt3204/envs/tftrain/bin/python"

# Read name + output_model from config
NAME=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['name'])")
OUTPUT_MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('output_model','transformer.pt'))")
DATE=$(date +%Y%m%d)
OUTPUT_DIR="${NAME}_${DATE}"

FORCE=false
[[ "$1" == "--force" ]] && FORCE=true

echo "Output dir   : $OUTPUT_DIR"
echo "Output model : $OUTPUT_DIR/$OUTPUT_MODEL"
echo "Force regen  : $FORCE"

# ── Job 1: Sample flux distributions ─────────────────────────────────────────
FLUX_CSV="$OUTPUT_DIR/flux_samples.csv"

if [[ "$FORCE" == false && -f "$FLUX_CSV" ]]; then
    echo "[Job 1] flux_samples.csv already exists, skipping."
    JOB1_ID=""
else
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
        --wrap="mkdir -p $OUTPUT_DIR/logs && $PYTHON $PAPER_DIR/sample_flux.py --config $CONFIG")
    echo "[Job 1] Submitted flux sampling: job $JOB1_ID"
fi

# ── Job 2: Generate labeling data (array) ────────────────────────────────────
N_SAMPLES=100000
CHUNK=5000
N_TASKS=$((N_SAMPLES / CHUNK))   # = 20

HDF5_CHECK="$OUTPUT_DIR/wb_model_labeling_data_0_${CHUNK}.h5"
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
        --output="$OUTPUT_DIR/logs/generate_%A_%a.out" \
        --error="$OUTPUT_DIR/logs/generate_%A_%a.err" \
        $DEPENDENCY_2 \
        --wrap="
mkdir -p $OUTPUT_DIR/logs
START_IDX=\$((SLURM_ARRAY_TASK_ID * $CHUNK))
END_IDX=\$((START_IDX + $CHUNK))
$PYTHON $PAPER_DIR/generate_training_data.py \
    --config $CONFIG \
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
    --output="$OUTPUT_DIR/logs/train_%j.out" \
    --error="$OUTPUT_DIR/logs/train_%j.err" \
    $DEPENDENCY_3 \
    --wrap="mkdir -p $OUTPUT_DIR/logs && $PYTHON $PAPER_DIR/train_ai_model.py --config $CONFIG --data-dir $OUTPUT_DIR")
echo "[Job 3] Submitted training: job $JOB3_ID"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $OUTPUT_DIR/logs/train_*.out"
