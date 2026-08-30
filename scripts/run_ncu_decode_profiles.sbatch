#!/bin/bash
#SBATCH --job-name=msa-ncu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=results/raw/ncu-%A_%a.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

case "${SLURM_ARRAY_TASK_ID:?missing Slurm array task ID}" in
  0) batch=1;  chunks=16; kernel=partial ;;
  1) batch=1;  chunks=16; kernel=merge ;;
  2) batch=16; chunks=4;  kernel=partial ;;
  3) batch=16; chunks=4;  kernel=merge ;;
  *) echo "unexpected array task ID: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

case "$kernel" in
  partial) kernel_regex='.*_gqa_sparse_decode_kernel.*' ;;
  merge) kernel_regex='.*_merge_topk_attn_out_kernel.*' ;;
esac

mkdir -p results/raw/ncu
commit="$(git rev-parse --short=12 HEAD)"
report_base="results/raw/ncu/msa-${commit}-b${batch}-c${chunks}-${kernel}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "commit=$(git rev-parse HEAD)"
echo "tag=$(git tag --points-at HEAD | paste -sd, -)"
echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "case=batch=${batch},chunks=${chunks},kernel=${kernel}"
nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,power.draw \
  --format=csv
ncu --version

ncu \
  --target-processes all \
  --profile-from-start off \
  --kernel-name-base function \
  --kernel-name "regex:${kernel_regex}" \
  --launch-count 1 \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --export "$report_base" \
  --force-overwrite \
  python benchmarks/profile_decode_kernel.py \
    --batch "$batch" \
    --chunks "$chunks" \
    --kernel "$kernel" \
    --warmup 20 \
    --profile-launches 1

echo "report=${report_base}.ncu-rep"
