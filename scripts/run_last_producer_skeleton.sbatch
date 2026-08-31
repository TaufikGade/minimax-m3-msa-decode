#!/bin/bash
#SBATCH --job-name=msa-last-producer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=results/raw/last-producer-skeleton-%A_%a.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

commit="$(git rev-parse --short=12 HEAD)"
case "${SLURM_ARRAY_TASK_ID}" in
  0) shape=tp1-scalar-b1-c16; groups=4; chunks=16; scale=scalar ;;
  1) shape=tp1-pertoken-b16-c4; groups=64; chunks=4; scale=per_token ;;
  2) shape=tp4-scalar-b1-c16; groups=1; chunks=16; scale=scalar ;;
  3) shape=tp4-pertoken-b16-c16; groups=16; chunks=16; scale=per_token ;;
  *) echo "unexpected task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python benchmarks/last_producer_skeleton.py \
  --groups "$groups" \
  --chunks "$chunks" \
  --scale-mode "$scale" \
  --graph-replays 1000 \
  --warmup 100 \
  --iterations 500 \
  --output "results/raw/last-producer-${commit}-${shape}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
