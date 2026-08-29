#!/bin/bash
#SBATCH --job-name=msa-splits
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/raw/split-sweep-%j.out

set -euo pipefail

repo_dir="$SLURM_SUBMIT_DIR"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python benchmarks/bench_split_sweep.py \
  --batches 1 4 8 16 \
  --chunks 1 2 4 8 16 \
  --warmup 20 \
  --iterations 100 \
  --output results/raw/split-sweep.csv
