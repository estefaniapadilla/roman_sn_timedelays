#!/bin/bash
#SBATCH --job-name=micro_lc
#SBATCH --account=m1727
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --array=0-19%20
# NOTE: mkdir -p logs before sbatch, or every task fails silently.
#SBATCH --output=logs/micro_lc_%A_%a.out
#SBATCH --error=logs/micro_lc_%A_%a.err
#SBATCH --licenses=scratch

set -euo pipefail

# Fresh env created by setup_env.sh (NOT sharma's venv)
source "${SCRATCH}/envs/slsim_micro/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

SCRIPT="${SLURM_SUBMIT_DIR}/extract_lightcurves.py"
RUN_DIR="${MICRO_RUN_DIR:-${SCRATCH}/lensed_sne_magmaps/run_002}"
OUTDIR="${MICRO_LC_OUTDIR:-${SCRATCH}/micro_lightcurves/run_002}"

mkdir -p "${OUTDIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"
echo "Run dir: ${RUN_DIR}"
echo "Out dir: ${OUTDIR}"
echo "Python:  $(command -v python)"

srun --cpu-bind=cores python -u "${SCRIPT}" \
    --run-dir "${RUN_DIR}" \
    --output-dir "${OUTDIR}" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --workers "${SLURM_CPUS_PER_TASK}"
