#!/bin/bash
#SBATCH --job-name=msa-cutlass-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/raw/cutlass-smoke-%j.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$repo_dir"
source "$repo_dir/.venv/bin/activate"

nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python - <<'PY'
import fmha_sm100
import torch

print(f"torch={torch.__version__}")
print(f"fmha_sm100={fmha_sm100.__file__}")
PY

python tests/smoke_cutlass.py --batch 1 --num-heads 16 --num-kv-heads 1
python tests/smoke_cutlass.py --batch 16 --num-heads 16 --num-kv-heads 1
python tests/smoke_cutlass.py --batch 1 --num-heads 64 --num-kv-heads 4
python tests/smoke_cutlass.py --batch 16 --num-heads 64 --num-kv-heads 4
