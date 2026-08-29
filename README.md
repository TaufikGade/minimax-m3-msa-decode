# MiniMax M3 MSA Small-Batch Decode

This repository investigates whether MiniMax M3 MSA small-batch decode on
NVIDIA B300 benefits from a dedicated kernel over the current Triton split-K
baseline. A measured result showing that the available speedup is smaller than
the implementation and maintenance cost is also a valid outcome.

## Scope

- Primary GPU: NVIDIA B300 (SM100)
- Batch sizes: 1, 4, 8, 16, with 32 as a crossover control
- Decode query length: 1
- Sparse shape: top-k 16, page size 128, head dimension 128
- Main KV-cache type: FP8
- Implementations: Triton baseline, forced CUTLASS comparison, and one bounded
  optimization challenge

The primary benchmark starts after top-k indices have been generated. Indexer
cost is reported separately.

## Team split

### Benchmark and correctness owner

- Build the standalone benchmark harness and FP32 reference.
- Maintain correctness cases and tolerances.
- Run benchmark matrices and CUDA Graph measurements.
- Independently validate optimized kernels and archive raw results.

### Kernel and profiling owner

- Analyze Triton PTX/SASS and collect Nsight Compute evidence.
- Study grid/wave behavior, occupancy, registers, memory traffic, and Tensor
  Core utilization.
- Run the bounded Triton configuration sweep.
- Implement a minimal optimization only when profiling shows credible headroom.

The Day 1 go/no-go review decides whether Day 2 is spent on implementation or
on completing the evidence chain for a negative result.

## Repository layout

~~~text
benchmarks/           benchmark entry points and profiling launchers
docs/                 task statement, experiment plan, and upstream notes
kernels/              project-owned Triton/CUDA experiments
results/raw/          machine-readable measurements (small files only)
results/figures/      plots used in the report and defense
scripts/              environment and reproducibility utilities
tests/                correctness tests and reference implementation
vendor/vllm_msa_ref/  pinned upstream vLLM source snapshot
~~~

## Upstream source

The directory vendor/vllm_msa_ref is a source snapshot of the MiniMax M3
implementation from vLLM commit d4da0c5. The files retain their original
Apache-2.0 SPDX and copyright headers.

See [the task statement](docs/TASK.md) and
[the two-day experiment plan](docs/EXPERIMENT_PLAN.md).

## Initial workflow

1. Create the repository-local environment:

   ~~~bash
   uv venv .venv \
     --python ~/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
   source .venv/bin/activate
   uv pip install -r requirements-b300.txt
   ~~~

2. Run scripts/collect_env.ps1 and archive the manifest with the raw data.
3. Implement and validate the FP32 reference before performance measurement.
4. Dry-run every command before reserving B300 time.
5. Measure run-to-run noise before claiming a speedup.
6. Use only same-machine B300 measurements in primary comparison figures.

The verified B300 environment uses Python 3.12.13, PyTorch 2.11.0+cu130,
Triton 3.6.0, and NumPy 2.5.2. The standalone baseline does not require a full
vLLM installation.

## License

This repository is licensed under Apache License 2.0. Vendored source remains
attributed to its original contributors.
