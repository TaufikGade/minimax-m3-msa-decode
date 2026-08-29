# Split-K Sweep Findings

## Experiment

- GPU: NVIDIA B300 SXM6 AC
- Slurm job: 11021
- KV cache: FP8 E4M3 with scalar K/V scales
- Shape: top-k 16, page size 128, head dimension 128
- Heads: 64 query heads / 4 KV heads
- Sequence length: 2048
- Batches: 1, 4, 8, 16
- Split counts: 1, 2, 4, 8, 16
- Samples per configuration: 100 after 20 warm-up launches
- Primary selection metric: CUDA Graph replay median

Every one of the 20 configurations passed the FP32-reference check. Maximum
absolute error was at most 0.00048828125 and cosine similarity was at least
0.9999948.

## Best split count

| Batch | Best chunks | Upstream chunks | Best Graph median | Same choice |
|---:|---:|---:|---:|:---:|
| 1 | 16 | 16 | 16.14 us | yes |
| 4 | 16 | 16 | 22.67 us | yes |
| 8 | 8 | 8 | 32.30 us | yes |
| 16 | 4 | 4 | 48.75 us | yes |

**Conclusion:** the pinned upstream split heuristic selected the lowest Graph
median at every tested batch. A split-count-only optimization has no measured
headroom in this matrix.

**Evidence:** all five legal power-of-two split counts were measured for every
batch using the same inputs and correctness reference.

**Limitation:** this conclusion applies to TP1-like 64/4 heads, FP8 scalar
scales, sequence length 2048, and the tested B300 software stack. TP4 and
per-token scale still require measurement.

## Partial and merge decomposition

For the upstream-default split count:

| Batch | Partial median | Merge median | Instrumented total | Merge fraction |
|---:|---:|---:|---:|---:|
| 1 | 31.02 us | 11.52 us | 42.67 us | 27.0% |
| 4 | 35.14 us | 9.34 us | 44.94 us | 20.8% |
| 8 | 43.07 us | 12.29 us | 55.34 us | 22.2% |
| 16 | 60.10 us | 12.32 us | 72.51 us | 17.0% |

**Conclusion:** merge is large enough to justify profiling a fusion strategy;
it is not yet evidence that fusion will recover the full measured fraction.

**Evidence:** the measured merge fraction is 17%-27%, above the experiment
plan's 15% investigation threshold for every default configuration.

**Limitation:** component timing inserts a CUDA Event between partial and merge.
The inserted event perturbs the two-kernel sequence, so the instrumented total
is higher than CUDA Graph replay. Treat the fraction as a prioritization signal,
not a direct end-to-end speedup bound.

## Split trade-off

Increasing chunks first reduces partial time by exposing more parallel work,
then increases merge work and intermediate traffic. The optimum moves from 16
chunks at batch 1/4 to 8 at batch 8 and 4 at batch 16. This is consistent with
the upstream target-grid heuristic: larger batches already provide enough grid
parallelism and need less split-K expansion.

## Next action

Do not spend challenge time tuning only the split count. Profile the default
batch 1 and batch 16 partial/merge kernels with Nsight Compute, focusing on:

- grid size and number of waves;
- registers and achieved occupancy;
- DRAM/L2 traffic from partial output and LSE;
- merge duration relative to Graph total;
- feasibility of reducing workspace traffic or merging within a persistent or
  cooperative design.

Only start a fusion prototype if profile evidence confirms that intermediate
traffic or the second launch is recoverable without reducing occupancy.
