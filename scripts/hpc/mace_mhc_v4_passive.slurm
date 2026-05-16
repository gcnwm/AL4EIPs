#!/bin/bash
#SBATCH --job-name=mace-mhc-v4-550
#SBATCH --partition=gpu4090
#SBATCH --qos=2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=1-3
#SBATCH --output=results/al_logs/mace_mhc_v4_550_%A_%a.out
#SBATCH --error=results/al_logs/mace_mhc_v4_550_%A_%a.err

# MACE-MHC passive 550-label baseline on the frozen v4 rMD17 split.
#
# Runs one seed per SLURM array task so the three seeds can execute in parallel
# if the scheduler permits it. This does NOT run active learning. It reuses the
# frozen v4 validation/test split and the passive 550-label seed datasets used
# by scripts/experiments/run_passive_v4_baseline.py; if the passive train XYZ is
# absent, the runner deterministically regenerates it from split_indices.npz.
#
# Usage:
#   sbatch scripts/hpc/mace_mhc_v4_passive.slurm
#
# Useful overrides:
#   PROJECT_ROOT=/gpfs/work/che/xinyangli22/fyp \
#   MACE_ENV=/gpfs/work/che/xinyangli22/.conda/env/mace \
#   MAX_EPOCHS=200 N_HEADS=4 N_TRAIN=550 \
#   sbatch scripts/hpc/mace_mhc_v4_passive.slurm
#
# Single seed without array semantics:
#   MHC_SEED=2 sbatch --array=1-1 scripts/hpc/mace_mhc_v4_passive.slurm
#
# Rerun/overwrite a completed seed directory:
#   FORCE=1 MHC_SEED=2 sbatch --array=1-1 scripts/hpc/mace_mhc_v4_passive.slurm

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/che/xinyangli22/fyp}"
MACE_ENV="${MACE_ENV:-/gpfs/work/che/xinyangli22/.conda/env/mace}"
N_TRAIN="${N_TRAIN:-550}"
N_HEADS="${N_HEADS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
EVALUATION_POLICY="${EVALUATION_POLICY:-fixed_member}"
RESULTS_ROOT="${RESULTS_ROOT:-results/mace_mhc_v4_550}"
FORCE="${FORCE:-0}"
SEED="${MHC_SEED:-${SLURM_ARRAY_TASK_ID:-1}}"

module load cuda/12.6.0
module load anaconda3
source activate "${MACE_ENV}"

cd "${PROJECT_ROOT}"
mkdir -p results/al_logs "${RESULTS_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

SEED_DIR="${RESULTS_ROOT}/mace_mhc_seed${SEED}"
METRICS="${SEED_DIR}/metrics.json"

if [[ -f "${METRICS}" && "${FORCE}" != "1" ]]; then
    echo "Metrics already exist for seed ${SEED}: ${METRICS}"
    echo "Set FORCE=1 to delete and rerun this seed."
    exit 0
fi

if [[ "${FORCE}" == "1" && -d "${SEED_DIR}" ]]; then
    echo "FORCE=1: removing existing seed directory ${SEED_DIR}"
    rm -rf "${SEED_DIR}"
fi

cat <<INFO
============================================================
MACE-MHC v4 passive 550 baseline
Job ID:        ${SLURM_JOB_ID:-N/A}
Array task:    ${SLURM_ARRAY_TASK_ID:-N/A}
Host:          $(hostname)
Project:       ${PROJECT_ROOT}
MACE env:      ${MACE_ENV}
Seed:          ${SEED}
Train labels:  ${N_TRAIN}
Heads:         ${N_HEADS}
Max epochs:    ${MAX_EPOCHS}
Eval policy:   ${EVALUATION_POLICY}
Results root:  ${RESULTS_ROOT}
Python:        $(which python)
GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)
Start:         $(date -Iseconds)
============================================================
INFO

start_seconds=${SECONDS}
python scripts/experiments/run_mace_mhc_v4_baseline.py \
    --seeds "${SEED}" \
    --n-train "${N_TRAIN}" \
    --n-heads "${N_HEADS}" \
    --max-epochs "${MAX_EPOCHS}" \
    --evaluation-policy "${EVALUATION_POLICY}" \
    --results-root "${RESULTS_ROOT}"
elapsed=$((SECONDS - start_seconds))

cat <<INFO
============================================================
Completed MACE-MHC seed ${SEED}
Elapsed:  $((elapsed / 3600))h $(((elapsed % 3600) / 60))m $((elapsed % 60))s
Metrics:  ${METRICS}
Finished: $(date -Iseconds)
============================================================
INFO
