# CUDA Graph Partial/Merge A/B Findings

## Scope and provenance

- Slurm job: 11864
- Measurement code commit: `341335ec31134d2fdca765994a328a87755c4b47`
- Baseline kernel tag: `baseline-b300-split-sweep-v1` at `617aa68`
- GPU: NVIDIA B300 SXM6 AC
- GPU UUID: `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`
- Driver: 580.126.09
- Shape: FP8 scalar scales, 64 query heads / 4 KV heads
- Cases: batch 1 / chunks 16 and batch 16 / chunks 4
- Independent paired trials: 10
- Per mode and trial: 100 warm-up and 500 measured Graph replays

Each fresh process captured three graphs over the same case and preallocated
workspace:

- `partial`: split-K partial kernel only;
- `merge`: merge kernel only, after populating valid partial/LSE inputs;
- `full`: partial followed by merge.

Mode order rotated across trials, and batch order alternated. All trials passed
the FP32-reference check.

## Results

| Batch | Partial | Merge only | Full | Paired `full - partial` | Eliminate-merge upper bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.262 us | 6.821 us | 10.280 us | 2.018 ± 0.205 us | 19.63% ± 2.04% |
| 16 | 24.413 us | 6.851 us | 28.496 us | 4.083 ± 0.017 us | 14.33% ± 0.05% |

The percentage is calculated per paired trial as:

\[
U = \frac{T_{full} - T_{partial}}{T_{full}}.
\]

It is deliberately named an upper bound. Removing the merge graph node also
removes work required for a correct output: cross-chunk LSE reduction and the
final output write. A fused implementation cannot remove that mathematical
work, and may pay additional synchronization or lose occupancy.

For batch 1, the per-trial upper bound ranges from 15.38% to 20.94%. For batch
16 it ranges from 14.27% to 14.38%. Both remain above the 10% go/no threshold,
and both are well above the previously established 5% significance threshold.

The machine-readable summary is
`results/raw/graph-ab-341335ec3113-11864.csv`, with SHA-256
`d26f0de49ddd5e810b5380db4ef51dcdb39a92823f681d6d696a4056bd38849f`.
The 20 input CSV files are under
`results/raw/graph-ab/msa-341335ec3113-11864/`. The Slurm log SHA-256 is
`c35197e54f2b8fdbb586726e549156f584038b7df79b16ef11856185bf49c7b9`.

## Interpretation with NCU

The A/B result and NCU evidence are consistent:

- merge workspace traffic is visible at DRAM;
- Graph execution still has a measurable incremental second-stage cost;
- the batch-1 partial grid has only 64 CTAs for 148 SMs;
- partial already uses high registers and shared memory, so adding reduction
  state to the same CTA may reduce residency or require a different grid.

The standalone merge-only median is about 6.8 us, but the incremental cost
inside the full graph is only 2.0–4.1 us because both standalone graphs include
common replay and scheduling costs. Therefore the earlier instrumented merge
fraction and merge-only duration must not be used directly as the fusion
speedup estimate.

## Go/no-go status

**TP1-like result:** go for a bounded fusion feasibility study, not yet for a
full implementation. The two evidence conditions are satisfied: workspace
traffic reaches DRAM, and the paired Graph upper bound exceeds 10%.

**Remaining gate:** repeat the fixed-shape controls for TP4-like heads and
per-token FP8 scales before choosing a fusion design. The current result does
not show that the same headroom exists when the partial grid is smaller, nor
that scale traffic leaves enough resource margin. Any prototype must be
compared against baseline in the same allocation and must exceed the 5%
stable-speedup threshold.
