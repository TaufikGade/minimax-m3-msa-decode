#!/bin/bash
#SBATCH --job-name=msa-cutlass-xover
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=results/raw/cutlass-crossover-%A_%a.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

commit="$(git rev-parse --short=12 HEAD)"
case "${SLURM_ARRAY_TASK_ID}" in
  0) num_heads=64; num_kv_heads=4; geometry=tp1 ;;
  1) num_heads=16; num_kv_heads=1; geometry=tp4 ;;
  *) echo "unexpected task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python benchmarks/bench_cutlass_decode.py \
  --batches 1 4 8 16 32 \
  --num-heads "$num_heads" \
  --num-kv-heads "$num_kv_heads" \
  --warmup 100 \
  --iterations 500 \
  --output "results/raw/cutlass-crossover-${commit}-${geometry}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
