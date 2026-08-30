#!/bin/bash
#SBATCH --job-name=msa-noise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/raw/noise-%j.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

trials="${TRIALS:-10}"
warmup="${WARMUP:-100}"
iterations="${ITERATIONS:-500}"
commit="$(git rev-parse --short=12 HEAD)"
output_dir="results/raw/noise/msa-${commit}-${SLURM_JOB_ID}"
summary="results/raw/run-noise-${commit}-${SLURM_JOB_ID}.csv"
mkdir -p "$output_dir"

echo "commit=$(git rev-parse HEAD)"
echo "baseline_commit=$(git rev-list -n 1 baseline-b300-split-sweep-v1)"
echo "job=$SLURM_JOB_ID"
echo "trials=$trials warmup=$warmup iterations=$iterations"
nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,power.draw \
  --format=csv

for trial in $(seq 1 "$trials"); do
  printf -v trial_id '%02d' "$trial"
  if (( trial % 2 == 1 )); then
    batches=(1 4 8 16)
  else
    batches=(16 8 4 1)
  fi
  echo "trial=$trial_id batches=${batches[*]}"
  python benchmarks/bench_decode.py \
    --batches "${batches[@]}" \
    --warmup "$warmup" \
    --iterations "$iterations" \
    --mode graph \
    --output "$output_dir/trial-${trial_id}.csv"
done

python benchmarks/summarize_run_noise.py \
  "$output_dir"/trial-*.csv \
  --output "$summary"

nvidia-smi --query-gpu=name,uuid,temperature.gpu,clocks.sm,power.draw \
  --format=csv
echo "summary=$summary"
