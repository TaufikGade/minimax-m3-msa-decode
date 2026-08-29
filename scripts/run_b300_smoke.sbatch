#!/bin/bash
#SBATCH --job-name=msa-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=results/raw/slurm-%j.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python tests/smoke_correctness.py
python benchmarks/bench_decode.py \
  --batches 1 4 8 16 \
  --warmup 20 \
  --iterations 100 \
  --mode both \
  --output results/raw/baseline-smoke.csv
