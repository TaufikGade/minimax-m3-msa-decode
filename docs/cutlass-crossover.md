# CUTLASS crossover measurement

This stage compares the pinned Triton split-K decode with the SM100 CUTLASS
decode path before any fusion prototype is considered.

## Frozen sources

- experiment baseline tag: `baseline-b300-split-sweep-v1` (`617aa68`)
- vLLM reference: `d4da0c55af3aa231b6209bf77871f3ed36eab0d2`
- vLLM MSA dependency: `087c161814d4d9c735b46c21212a09e5f8eb92fa`
- MSA CUTLASS submodule: `eb61c911471867a5fd2466bfd8f29306cea6ebf8`
- CUTLASS DSL: `4.6.0`
- quack-kernels: `0.6.1`

Install the optional environment on top of the B300 baseline environment:

```bash
uv pip install -r requirements-cutlass-b300.txt
```

The MSA source dependency must include its `python/fmha_sm100/cutlass`
submodule. A filtered manual clone therefore needs an explicit recursive
submodule update before installation.

## Equivalence controls

Both backends use the same BF16 query values, paged FP8 K/V cache, shuffled
physical page table, scalar K/V scales, sequence length 2048, page size 128,
and 16 selected logical blocks. CUTLASS consumes an FP8-quantized copy of Q
and receives its dequantization scale. Its strictly-ascending sparse-index
contract is enforced by sorting logical top-k indices; the Triton comparison
uses that same order.

The smoke test checks both backends against the FP32 gathered-KV reference.
The looser CUTLASS tolerance accounts for its extra FP8 Q quantization.

## Measurement boundary

`benchmarks/bench_cutlass_decode.py` records the following independently for
eager execution and CUDA Graph replay:

- `cutlass/metadata`: runtime KV lengths and causal offsets update;
- `cutlass/attention`: `fmha_sm100` sparse decode call only;
- `cutlass/full`: metadata update plus attention;
- `triton/full`: pinned split-K partial plus LSE merge on the same input.

Planning, allocation, JIT compilation, reference calculation, and graph
capture are outside timed regions. CUDA events report device-stream elapsed
time; they do not represent Python wall-clock dispatch latency.

Run the two geometries through Slurm:

```bash
sbatch scripts/run_cutlass_crossover.sbatch
```

The array tasks cover batches 1, 4, 8, 16, and 32 for TP1-like 64/4 heads and
TP4-like 16/1 heads, with 100 warmups and 500 timed iterations per row.
