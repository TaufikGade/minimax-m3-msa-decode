# Benchmarks

Place standalone decode benchmarks and bounded configuration sweeps here.
Every benchmark should support a fixed random seed, machine-readable output,
warm-up control, and preallocated workspaces.

Primary results must distinguish eager launch, CUDA Graph replay, partial
kernel time, merge time, and metadata overhead.

Current entry points:

- bench_decode.py: upstream-heuristic eager and CUDA Graph baseline.
- bench_split_sweep.py: correctness-checked partial/merge decomposition and
  power-of-two split-count sweep.
