# Per-Token FP8 Scale Control Findings

## Scope and correctness

- Code commit: `1410e42aaeac32704696d1ea8522f6d1b37102c9`
- GPU: NVIDIA B300 SXM6 AC, UUID
  `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`
- Graph jobs: TP1-like 11896, TP4-like 11897
- NCU jobs: per-token 11898/11899, same-commit scalar 11907/11908
- Graph protocol: ten trials, 100 warmups, 500 samples per mode
- Scale layout: `[num_kv_heads, physical_pages * 128]`

The scale tensors contain varying positive values rather than a repeated
constant. Quantization, the Triton kernel, and the FP32 reference all index
scales by physical page and token. This makes the check sensitive to the
two-level `topk_idx -> block_table -> physical page` mapping.

All cases passed the FP32-reference check. Per-token maximum absolute error was
at most 0.00048828125 and cosine similarity was at least 0.9999932.

## CUDA Graph comparison

Values below are per-token minus scalar. The two scale modes were alternated
within each trial on the same GPU allocation.

| Shape | Batch | Partial delta | Full replay delta | Per-token eliminate-merge upper bound |
|---|---:|---:|---:|---:|
| TP1 | 1 | +1.645 us (+19.61%) | +1.800 us (+17.50%) | 17.00% |
| TP1 | 8 | -2.032 us (-12.54%) | -2.042 us (-10.05%) | 22.39% |
| TP1 | 16 | -2.054 us (-8.42%) | -2.042 us (-7.17%) | 15.48% |
| TP4 | 1 | +0.030 us (+0.36%) | +0.029 us (+0.29%) | 16.29% |
| TP4 | 8 | +0.021 us (+0.21%) | +0.040 us (+0.33%) | 16.95% |
| TP4 | 16 | -1.981 us (-16.42%) | 0.000 us (0.00%) | 28.73% |

Using the previously measured 5% significance floor, TP1 changes are material
at all three batches. TP4 batch 1 and 8 are unchanged; TP4 batch 16 has a
material partial-only speedup that does not reduce full replay time. This last
case also shows why `full - partial` is only a loose fusion upper bound: PDL and
the dependent merge launch change the incremental cost even when standalone
partial time falls.

Machine-readable results are in `results/raw/graph-scale-summary.csv`. The four
source summaries and 120 trial CSVs are archived under `results/raw/`.

## NCU comparison

| Shape | Batch | NCU duration delta | DRAM-byte delta | Registers | Dynamic smem | Waves/SM | Issue active |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP1 | 1 | +7.26% | +3.45% | 195 -> 255 | 73.7 -> 42.0 KiB | 0.22 -> 0.22 | 23.05% -> 27.38% |
| TP1 | 16 | -10.83% | +3.12% | 168 -> 255 | 73.7 -> 42.0 KiB | 0.58 -> 0.86 | 34.30% -> 48.27% |
| TP4 | 1 | -0.24% | +4.33% | 195 -> 255 | 73.7 -> 42.0 KiB | 0.05 -> 0.05 | 22.47% -> 27.83% |
| TP4 | 16 | -8.42% | +3.19% | 168 -> 255 | 73.7 -> 42.0 KiB | 0.58 -> 0.86 | 30.19% -> 41.54% |

Per-token scale adds only about 3%-4% to observed DRAM bytes. DRAM remains far
from peak (0.58%-11.45%), so bandwidth is not the governing effect. Instead,
the compile-time `KV_SCALE_MODE` specialization changes the generated resource
shape: register use reaches 255/thread while dynamic shared memory falls by
43%. There is no Triton autotune decorator on this kernel. At batch 16 the
smaller shared-memory footprint raises waves/SM and issue activity enough to
outweigh extra scale work; at batch 1 the grid remains underfilled, so that
benefit cannot be realized.

Machine-readable metrics and report hashes are in
`results/raw/ncu-scale-summary.csv`. Raw `.ncu-rep` files remain uncommitted.

## Fusion decision

Per-token scale does not remove the latency upper bound: every tested shape
still has a per-token eliminate-merge bound above 15%. It does remove the
assumption that the current partial kernel has comfortable register headroom.
The per-token specialization already uses 255 registers/thread, so adding
merge state directly to that program risks spills or another resource-shape
change.

Therefore the evidence supports continued fusion design work, but not an
immediate fused-kernel implementation. The next bounded step is to specify how
cross-CTA split reduction would work and compile a resource-only skeleton for
both scale modes. A prototype should proceed only if that skeleton avoids local
spills and preserves the batch-16 wave count; TP4 batch 1 must remain a required
acceptance shape.
