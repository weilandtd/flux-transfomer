#!/bin/bash
# 13C refit with the published MFA fluxes pinned.
#
# Usage:
#   bash paper/submit/submit_experimental_mid_mfa.sh [models]
#   MODELS=e_coli_core bash paper/submit/submit_experimental_mid_mfa.sh
#
# [models]  optional comma-separated model keys from paper/scripts/_figure_common.py
#           (default: every model with an experimental panel)
#
# Needs <run-dir>/experimental_mid/tokens.csv (submit_experimental_mid.sh) and writes
# <run-dir>/experimental_mid_mfa/{tokens,fit,fluxes}.csv per model. One Ipopt fit per growth
# condition on the genome-scale networks; submit one job per organism to run them in
# parallel.
#
# Environment knobs:
#   MODELS=<keys>       comma-separated model keys, same as --models
#   RUN_SUFFIX=<date>   plot a different set of runs without editing _figure_common.py
#   DRY_RUN=1           print the sbatch command, submit nothing
#   CPUS, MEM, TIME, QOS, CONDA_ENV   override the defaults

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/paper/submit/_common.sh"

MODELS="${1:-${MODELS:-}}"

CPUS="${CPUS:-2}"
MEM="${MEM:-10G}"
# Requested walltime <= 1h routes to QOS=test (2 concurrent jobs); stay above it.
TIME="${TIME:-16:00:00}"
QOS="${QOS:-short}"

LOG_DIR="$REPO_ROOT/paper/.results/logs"
mkdir -p "$LOG_DIR"

echo "======================================"
echo "SigmaFlux — 13C refit with the published MFA fluxes pinned"
echo "Models     : ${MODELS:-all}"
echo "Output     : <run-dir>/experimental_mid_mfa/tokens.csv per model"
echo "Resources  : ${CPUS} cpus, ${MEM}, ${TIME}, qos=${QOS}"
echo "======================================"

BODY="$(slurm_preamble)
cd $REPO_ROOT
python -u paper/scripts/experimental_mid_mfa.py${MODELS:+ \\
    --models $MODELS}${RUN_SUFFIX:+ \\
    --run-suffix $RUN_SUFFIX}"

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY_RUN — would submit:"
    echo "$BODY"
    exit 0
fi

JOB_ID=$(sbatch --parsable \
    --job-name=exp_mid_mfa \
    --nodes=1 --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    ${QOS:+--qos=$QOS} \
    ${DEPENDENCY:+--dependency=afterok:$DEPENDENCY} \
    --output="$LOG_DIR/exp_mid_mfa_%j.out" \
    --error="$LOG_DIR/exp_mid_mfa_%j.err" \
    --wrap="$BODY")

echo "Submitted  : job $JOB_ID"
echo ""
echo "Monitor : squeue -j $JOB_ID"
echo "Logs    : $LOG_DIR/exp_mid_mfa_${JOB_ID}.out"
