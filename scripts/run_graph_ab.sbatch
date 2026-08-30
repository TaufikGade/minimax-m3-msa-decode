#!/bin/bash
#SBATCH --job-name=msa-graph-ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/raw/graph-ab-%j.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

trials="${TRIALS:-10}"
warmup="${WARMUP:-100}"
iterations="${ITERATIONS:-500}"
num_heads="${NUM_HEADS:-64}"
num_kv_heads="${NUM_KV_HEADS:-4}"
IFS=: read -r -a configured_batches <<< "${BATCHES:-1:16}"
commit="$(git rev-parse --short=12 HEAD)"
shape="h${num_heads}-kv${num_kv_heads}"
output_dir="results/raw/graph-ab/msa-${commit}-${shape}-${SLURM_JOB_ID}"
summary="results/raw/graph-ab-${commit}-${shape}-${SLURM_JOB_ID}.csv"
mkdir -p "$output_dir"

echo "commit=$(git rev-parse HEAD)"
echo "baseline_commit=$(git rev-list -n 1 baseline-b300-split-sweep-v1)"
echo "job=$SLURM_JOB_ID"
echo "trials=$trials warmup=$warmup iterations=$iterations"
echo "num_heads=$num_heads num_kv_heads=$num_kv_heads batches=${configured_batches[*]}"
nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,power.draw \
  --format=csv

for trial in $(seq 1 "$trials"); do
  printf -v trial_id '%02d' "$trial"
  case $((trial % 3)) in
    1) modes=(partial full merge) ;;
    2) modes=(full merge partial) ;;
    0) modes=(merge partial full) ;;
  esac
  if (( trial % 2 == 1 )); then
    batches=("${configured_batches[@]}")
  else
    batches=()
    for ((index=${#configured_batches[@]}-1; index>=0; index--)); do
      batches+=("${configured_batches[index]}")
    done
  fi
  for batch in "${batches[@]}"; do
    echo "trial=$trial_id batch=$batch modes=${modes[*]}"
    python benchmarks/bench_graph_ab.py \
      --batch "$batch" \
      --num-heads "$num_heads" \
      --num-kv-heads "$num_kv_heads" \
      --warmup "$warmup" \
      --iterations "$iterations" \
      --order "${modes[@]}" \
      --output "$output_dir/trial-${trial_id}-b${batch}.csv"
  done
done

python benchmarks/summarize_graph_ab.py \
  "$output_dir"/trial-*.csv \
  --output "$summary"

nvidia-smi --query-gpu=name,uuid,temperature.gpu,clocks.sm,power.draw \
  --format=csv
echo "summary=$summary"
