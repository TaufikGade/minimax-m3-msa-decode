# CUTLASS sparse-decode crossover findings

## Provenance

- measurement commit: `fc8160616c7a`
- Slurm array: `12376`
- GPU: NVIDIA B300 SXM6 AC
- samples per row: 100 warmups, 500 timed iterations
- shapes: batch 1/4/8/16/32, sequence length 2048, top-k 16, page 128
- TP1-like geometry: 64 query heads / 4 KV heads
- TP4-like geometry: 16 query heads / 1 KV head

The raw data are in:

- `results/raw/cutlass-crossover-fc8160616c7a-tp1-12376_0.csv`
- `results/raw/cutlass-crossover-fc8160616c7a-tp4-12376_1.csv`

## Correctness control

The CUTLASS path consumes an FP8-quantized copy of the same underlying query
used by the BF16 Triton path, matching the upstream wrapper contract. Both use
the same paged FP8 KV data, scalar KV scales, logical sparse blocks, and
shuffled physical page table.

Across all measured shapes, CUTLASS versus the FP32 gathered-KV reference had
cosine similarity 0.99967--0.99972 and maximum absolute error no greater than
0.00281. Triton cosine similarity was above 0.99999.

## CUDA Graph full-path crossover

Median device-stream time in microseconds:

| Geometry | Batch | CUTLASS full | Triton full | CUTLASS relative to Triton |
|---|---:|---:|---:|---:|
| TP1 64/4 | 1 | 30.11 | 15.94 | +88.96% |
| TP1 64/4 | 4 | 32.13 | 21.89 | +46.78% |
| TP1 64/4 | 8 | 32.14 | 32.13 | +0.05% |
| TP1 64/4 | 16 | 32.16 | 48.54 | -33.75% |
| TP1 64/4 | 32 | 36.26 | 77.22 | -53.05% |
| TP4 16/1 | 1 | 30.08 | 15.78 | +90.67% |
| TP4 16/1 | 4 | 32.13 | 17.79 | +80.58% |
| TP4 16/1 | 8 | 31.90 | 17.79 | +79.32% |
| TP4 16/1 | 16 | 32.13 | 21.89 | +46.78% |
| TP4 16/1 | 32 | 32.13 | 32.13 | +0.00% |

Negative relative values mean CUTLASS is faster. Graph within-run CV is below
3% for all full-path rows. The TP1 crossover is near batch 8, and CUTLASS has a
clear advantage from batch 16. TP4 only reaches parity at batch 32; no tested
TP4 batch has a stable CUTLASS advantage.

The upstream common cutoff of batch 16 is therefore supported for TP1 but not
for TP4. A geometry-aware policy should keep TP4 on Triton at batch 16 and
must measure batches above 32 before selecting a CUTLASS threshold for TP4.

## Eager result

CUDA-event eager medians for CUTLASS and Triton are within about 3% through
batch 8 for TP1 and through batch 32 for TP4. Those small differences are below
the stability rule because within-run CV is roughly 5--8%. Only the TP1 wins at
batch 16 (-17.22%) and batch 32 (-53.70%) clearly exceed that noise bound.

This event measurement reports device-stream elapsed time, not Python
wall-clock dispatch latency. The Graph results are the stronger crossover
evidence for the serving path.

## Metadata cost

The metadata-only Graph median is about 7.0--7.6 us, but it is not additive to
the attention-only time because an isolated one-kernel Graph includes fixed
replay/event costs. The credible incremental metadata cost is the paired
`CUTLASS full - CUTLASS attention` difference in the same run: 0.03--2.05 us
for the tested Graph shapes.

Consequently the metadata update is not the dominant CUTLASS cost and removing
it cannot change the TP4 batch-16 decision. The approximately 30--34 us
CUTLASS attention floor dominates small batches.

## Independent boundary runs

Slurm array `12386` ran each boundary shape in 10 independent Python
processes. Every process rebuilt the random input with a distinct seed and
performed 100 warmups followed by 500 timed iterations. The measurement code
commit was `a51c14620c2e`.

| Geometry | Batch | CUTLASS mean median | Triton mean median | Paired delta | CUTLASS run CV | Triton run CV |
|---|---:|---:|---:|---:|---:|---:|
| TP1 64/4 | 8 | 20.099 us | 20.106 us | -0.03% | 0.25% | 0.24% |
| TP1 64/4 | 16 | 20.157 us | 28.352 us | -28.91% | 0.05% | 0.00% |
| TP4 16/1 | 32 | 20.154 us | 20.160 us | -0.03% | 0.06% | 0.00% |
| TP4 16/1 | 64 | 20.147 us | 28.352 us | -28.94% | 0.08% | 0.00% |

The summary is in `results/raw/cutlass-boundary-noise-summary.csv`; the 40
per-run CSV files retain every component and execution mode. Some medians land
on the same CUDA-event timing quantum, explaining zero reported run CV in two
Triton rows.

Absolute medians in these dedicated runs differ from the earlier full sweep,
consistent with a different GPU clock/runtime state. The paired crossover
conclusion is unchanged. Applying `Delta > max(5%, 2 CV)`, TP1 batch 16 and
TP4 batch 64 are stable CUTLASS wins, while TP1 batch 8 and TP4 batch 32 are
ties.

The conservative tested dispatch policy is therefore CUTLASS from batch 16
for TP1 and from batch 64 for TP4. The exact TP4 transition within batches
33--63 remains unmeasured.

## Implication for fusion go/no-go

These results do not justify writing a fusion kernel yet. They provide a
backend crossover boundary, not evidence that Triton's partial/merge fusion is
profitable. The existing NCU evidence about merge traffic, launch cost,
register pressure, and occupancy remains the gate for fusion. In particular:

- use Triton for latency-sensitive small batches;
- CUTLASS is already a strong alternative for TP1 batch 16 and above;
- do not generalize the TP1 batch-16 cutoff to TP4;
- the tested conservative TP4 cutoff is batch 64, not batch 16;
- measure intermediate TP4 batches only if an exact 33--63 transition matters
  to the production batch distribution.
