#!/bin/bash
#SBATCH --job-name=msa-ncu-tp4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=results/raw/ncu-tp4-%A_%a.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

case "${SLURM_ARRAY_TASK_ID:?missing Slurm array task ID}" in
  0) batch=1;  kernel=partial ;;
  1) batch=1;  kernel=merge ;;
  2) batch=16; kernel=partial ;;
  3) batch=16; kernel=merge ;;
  *) echo "unexpected array task ID: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

case "$kernel" in
  partial) kernel_regex='.*_gqa_sparse_decode_kernel.*' ;;
  merge) kernel_regex='.*_merge_topk_attn_out_kernel.*' ;;
esac

num_heads=16
num_kv_heads=1
chunks=16
scale_mode="${SCALE_MODE:-scalar}"
mkdir -p results/raw/ncu
commit="$(git rev-parse --short=12 HEAD)"
report_base="results/raw/ncu/msa-${commit}-tp4-s${scale_mode}-b${batch}-c${chunks}-${kernel}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "commit=$(git rev-parse HEAD)"
echo "baseline_commit=$(git rev-list -n 1 baseline-b300-split-sweep-v1)"
echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "case=tp4,batch=${batch},chunks=${chunks},kernel=${kernel},scale_mode=${scale_mode}"
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
    --num-heads "$num_heads" \
    --num-kv-heads "$num_kv_heads" \
    --scale-mode "$scale_mode" \
    --warmup 20 \
    --profile-launches 1

echo "report=${report_base}.ncu-rep"
