# Initial Nsight Compute Findings

## Scope and provenance

- GPU: NVIDIA B300 SXM6 AC, 148 SMs, compute capability 10.3
- Driver: 580.126.09
- Nsight Compute: 2025.3.1.0
- Code commit: `9eb1d6e6e8011d1f5b3f93622fb99a3b507b7f76`
- Baseline tag: `baseline-b300-split-sweep-v1` at commit `617aa68`
- Shape: FP8 scalar scales, random physical pages, 64 query heads / 4 KV heads
- Profiled cases: batch 1 / chunks 16 and batch 16 / chunks 4
- NCU sections: LaunchStats, Occupancy, SpeedOfLight,
  MemoryWorkloadAnalysis, SchedulerStats, and WarpStateStats

All four cases passed the FP32-reference check. The maximum absolute error was
at most 0.00048828125 and cosine similarity was at least 0.9999952.

The raw `.ncu-rep` files are intentionally not committed. Their SHA-256 hashes
are:

| Case | Slurm job | SHA-256 |
|---|---:|---|
| batch 1 partial | 11854_0 | `a353ce2e4606dc7fa187aad2b77bbcdf6416f932c3285851802cd914917003ca` |
| batch 1 merge | 11855_1 | `a1c89acde3502221ac7c4750006d3c6b1182fd0741629110f5790777b9e9b711` |
| batch 16 partial | 11855_2 | `5790c205c95bb2816e00411cf19c619839029ed23d989bc462b52628f4bcec82` |
| batch 16 merge | 11855_3 | `cab6fc477a34300ac76be6dd2cde069eefa6ed7ca667cf42f8bf320310c546d2` |

Machine-readable values are stored in `results/raw/ncu-summary.csv`.

## Summary

| Batch | Kernel | CTAs | Reg/thread | Dynamic smem | NCU duration | Estimated DRAM bytes | DRAM peak | Tensor pipe | Achieved occupancy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | partial | 64 | 195 | 73.7 KiB | 13.60 us | 2.053 MiB | 2.07% | 1.50% | 6.25% |
| 1 | merge | 64 | 32 | 2.0 KiB | 7.78 us | 0.260 MiB | 0.46% | 0% | 6.19% |
| 16 | partial | 256 | 168 | 73.7 KiB | 44.29 us | 32.294 MiB | 9.98% | 7.52% | 11.26% |
| 16 | merge | 1024 | 22 | 2.0 KiB | 9.73 us | 1.021 MiB | 1.44% | 0% | 38.83% |

`Estimated DRAM bytes` is derived from NCU's DRAM bytes/second multiplied by
NCU duration. It is useful for traffic accounting but is not an independently
collected byte counter.

## Partial kernel

**Conclusion:** batch 1 is primarily underfilled at the whole-GPU level, while
high register and shared-memory usage also bound the number of resident CTAs.

**Evidence:** batch 1 launches only 64 CTAs on 148 SMs. Each thread uses 195
registers and each CTA uses 73.7 KiB dynamic shared memory. NCU reports a
register limit of two resident CTAs per SM, 12.5% theoretical occupancy, and
6.25% achieved occupancy. Tensor-pipe activity is only 1.50%, DRAM reaches only
2.07% of peak, and the scheduler has an eligible warp in 23.14% of cycles.

At batch 16 the grid grows to 256 CTAs. Tensor-pipe activity rises to 7.52%,
DRAM throughput to 9.98% of peak, and scheduler issue-active to 34.54%. This is
still not a simple bandwidth-saturation regime. The measured 32.294 MiB of
DRAM traffic is close to the expected order of magnitude for reading FP8 K/V
for 16 requests plus writing the split workspace.

**Limitation:** low achieved occupancy alone does not prove that reducing
registers will improve latency. Batch 1 cannot fill all SMs even if each CTA
uses fewer registers. A register-reduction experiment would need to preserve
the generated instruction count and shared-memory behavior.

## Merge kernel and workspace traffic

**Conclusion:** the split workspace reaches DRAM, but merge is not a saturated
DRAM-bandwidth kernel. Its cost combines a short reduction with limited grid
parallelism and fixed launch/scheduling overhead.

**Evidence:** the theoretical workspace sizes are:

- batch 1 / chunks 16: 266,240 bytes of partial output plus LSE;
- batch 16 / chunks 4: 1,064,960 bytes of partial output plus LSE.

Adding the final output write gives theoretical merge traffic of 282,624 bytes
for batch 1 and 1,327,104 bytes for batch 16. NCU estimates 273,152 bytes and
1,070,080 bytes respectively. The close batch-1 match and the same order of
magnitude at batch 16 support the claim that workspace traffic is visible at
DRAM; cache residency likely explains part of the batch-16 difference.

Despite that traffic, merge reaches only 0.46% and 1.44% of peak DRAM
throughput. Batch 1 launches only 64 CTAs, and NCU reports no eligible warp in
92.30% of scheduler cycles. Batch 16 launches 1024 short CTAs and reaches
38.83% achieved occupancy, but still has no eligible warp in 81.49% of cycles.

**Limitation:** the NCU durations include profiler replay effects and must not
replace the CUDA Graph latency measurements. They describe device behavior,
not production end-to-end latency.

## Fusion go/no-go status

The evidence satisfies one prerequisite for investigating fusion: partial/LSE
workspace traffic is visible at DRAM. It does not yet satisfy a go decision.

An idealized fusion that removes both the workspace write and read would avoid
about 0.508 MiB at batch 1 and 2.031 MiB at batch 16. Relative to the combined
NCU-estimated DRAM traffic, these are upper bounds of approximately 21.9% and
6.1%. They are traffic bounds, not latency bounds: the reduction and final
output write remain, and any cooperative or persistent design may reduce grid
parallelism or occupancy.

Before implementing fusion, perform a CUDA Graph A/B experiment that isolates
the recoverable second-launch cost, and add the TP4-like and per-token-scale
controls. The current evidence supports continued measurement, not immediate
kernel implementation.
