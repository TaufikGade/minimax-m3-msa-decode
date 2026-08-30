# TP4-Like Scalar-Scale Control Findings

## Scope

- Code commit: `ac55b6fdd5095d44741d1461f25f8615f0789f95`
- GPU: NVIDIA B300 SXM6 AC
- GPU UUID: `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`
- Heads: 16 query / 1 KV
- KV cache: FP8 E4M3 with scalar K/V scales
- Default chunks: 16 for batch 1, 8, and 16
- Graph A/B job: 11871, ten paired trials, 500 samples per mode
- NCU jobs: 11875_0 and 11876_1-3

All cases passed the FP32-reference check. Maximum absolute error was
0.00048828125 and cosine similarity was at least 0.9999946.

## CUDA Graph upper bound

| Batch | Partial | Full | Paired `full - partial` | Eliminate-merge upper bound |
|---:|---:|---:|---:|---:|
| 1 | 8.091 us | 10.061 us | 1.970 ± 0.076 us | 19.58% ± 0.76% |
| 8 | 10.043 us | 12.096 us | 2.053 ± 0.011 us | 16.97% ± 0.09% |
| 16 | 12.061 us | 14.144 us | 2.083 ± 0.024 us | 14.73% ± 0.16% |

All three loose upper bounds exceed both the 10% feasibility threshold and the
5% stable-speedup threshold. TP4 therefore does not invalidate the bounded
fusion feasibility study. These remain impossible-best-case bounds because a
correct fused path must still perform reduction, synchronization, and the final
output write.

The formal summary is
`results/raw/graph-ab-ac55b6fdd509-h16-kv1-11871.csv` (SHA-256
`e226a2ed3d1b973bfc25cd8ad0ab02ae21362bc643374cea22777ad7e91169c6`).
The 30 input CSVs are under
`results/raw/graph-ab/msa-ac55b6fdd509-h16-kv1-11871/`.

## NCU results

| Batch | Kernel | CTAs | Reg/thread | Dynamic smem | NCU duration | Estimated DRAM bytes | Tensor pipe | Occupancy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | partial | 16 | 195 | 73.7 KiB | 13.09 us | 0.541 MiB | 0.38% | 6.15% |
| 1 | merge | 16 | 32 | 2.0 KiB | 6.66 us | 0.070 MiB | 0% | 6.55% |
| 16 | partial | 256 | 168 | 73.7 KiB | 18.08 us | 8.104 MiB | 4.56% | 10.98% |
| 16 | merge | 256 | 32 | 2.0 KiB | 8.03 us | 1.022 MiB | 0% | 10.86% |

At batch 1 both stages launch only 16 CTAs on 148 SMs. The partial kernel keeps
the same 195 registers/thread and 73.7 KiB shared-memory footprint seen in the
TP1-like batch-1 case, so reducing head count makes whole-GPU underfill more
severe without relieving per-CTA resource pressure.

The DRAM estimates agree with shape accounting. Batch-1 merge theoretically
reads 66,560 bytes of partial/LSE workspace and writes 4,096 output bytes;
NCU estimates 73,472 bytes. At batch 16, `chunks * query_heads` equals the TP1
control's value, so the workspace size remains about 1.02 MiB and is again
visible at DRAM.

The raw `.ncu-rep` files remain uncommitted. Their checksums are:

| Case | SHA-256 |
|---|---|
| batch 1 partial | `e43e1373d2e4c754453aba4b49b5b9abe92c5a8d420a1db4e3e601ba1f082b94` |
| batch 1 merge | `f413324159d6cd9d3761026508991dafe61bc14e684d455908e40dd6982dc499` |
| batch 16 partial | `dfc906ac267bccd0d33faccd2417180b1573b5287f3640382fa292b15a73823a` |
| batch 16 merge | `b201ea2426d25b0cd0b93afe1df384bb54661eb167098d47132d4a5fe71a726c` |

Machine-readable metrics are in `results/raw/ncu-tp4-summary.csv`.

## Decision

The scalar-scale evidence now supports a bounded feasibility study for both
TP1-like and TP4-like heads. TP4 batch 1 is the harshest parallelism case and
should be a required prototype acceptance shape. Per-token FP8 scale remains
the final control gate before selecting a design; no scalar-only result should
be generalized to that scale layout.
