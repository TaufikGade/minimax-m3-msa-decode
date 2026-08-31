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
- profile_decode_kernel.py: validate and warm up one fixed shape, then launch
  only the selected partial or merge kernel inside a CUDA profiler range. Use
  it with `ncu --profile-from-start off` so setup and warm-up are not captured.
- summarize_run_noise.py: aggregate independent Graph benchmark runs, compute
  CV across per-run medians, and report the `max(5%, 2CV)` significance
  threshold. `scripts/run_b300_noise.sbatch` generates the input trials.
- bench_graph_ab.py and summarize_graph_ab.py: measure paired partial-only,
  merge-only, and full Graph replay latency. The `full - partial` difference is
  a deliberately loose upper bound on eliminating the second stage, not an
  achievable fusion-speedup claim. `scripts/run_graph_ab.sbatch` runs trials
  and accepts `NUM_HEADS`, `NUM_KV_HEADS`, `SCALE_MODE`, and colon-separated
  `BATCHES` environment overrides for TP and FP8-scale controls. Use
  `SCALE_MODES=scalar:per_token_head` to alternate both modes within one GPU
  allocation. Per-token scales use the upstream-compatible shape
  `[num_kv_heads, physical_pages * 128]`.
- bench_cutlass_decode.py accepts `--effective-kv-len` while retaining the
  fixed 2048-token allocation and 16-page top-k capacity. This isolates runtime
  KV length from layout/capacity changes. `scripts/run_cutlass_kvlen_crossover.sbatch`
  measures 128/512/1024/2048 tokens at the TP1 batch 8/16 and TP4 batch 32/64
  boundaries in ten independent processes per shape. Aggregate its CSV files
  with summarize_cutlass_boundary_noise.py; older CSV files without the new
  column are treated as 2048-token cases.
