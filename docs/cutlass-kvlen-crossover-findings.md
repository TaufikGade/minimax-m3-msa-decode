# CUTLASS effective-KV-length crossover findings

## Provenance

- measurement commit: `3dbb4e609578`
- Slurm array: `13977`
- GPU: NVIDIA B300 SXM6 AC
- workload: one-token decode, scalar FP8 KV scales, top-k 16, page size 128
- allocation capacity: fixed at 2048 tokens / 16 logical pages
- effective KV lengths: 128, 512, 1024, 2048 tokens
- boundary shapes: TP1-like batch 8/16 and TP4-like batch 32/64
- sampling: 10 independent Python processes per shape, each with 100 warmups
  and 500 timed iterations

Only the runtime `seq_lens` and CUTLASS plan length change. KV allocation,
physical-page permutation, top-k capacity, and benchmark procedure stay fixed,
so this isolates effective KV length from capacity and layout changes.

The aggregate data are in
`results/raw/cutlass-kvlen-summary-13977.csv`. The 160 per-run CSV files retain
component-level eager and CUDA Graph measurements, accuracy, seed, GPU, and
commit provenance.

## Correctness

Before the sweep, B300 smoke tests covered both geometries at effective lengths
128 and 1024. Triton and CUTLASS both passed comparison with the FP32 gathered-KV
reference. CUTLASS cosine similarity was 0.99971--0.99973; Triton cosine
similarity exceeded 0.99999. Every sweep process also performed the same
correctness checks before timing, and no array task reported an error.

## CUDA Graph full-path results

Mean of the ten independent per-run medians, in microseconds. A negative delta
means CUTLASS is faster.

| Geometry | Batch | Effective KV | CUTLASS | Triton | CUTLASS vs Triton |
|---|---:|---:|---:|---:|---:|
| TP1 64/4 | 8 | 128 | 24.003 | 19.888 | +20.69% |
| TP1 64/4 | 8 | 512 | 28.077 | 32.176 | -12.74% |
| TP1 64/4 | 8 | 1024 | 32.154 | 32.154 | 0.00% |
| TP1 64/4 | 8 | 2048 | 32.176 | 32.179 | -0.01% |
| TP1 64/4 | 16 | 128 | 26.019 | 23.858 | +9.06% |
| TP1 64/4 | 16 | 512 | 28.083 | 46.518 | -39.63% |
| TP1 64/4 | 16 | 1024 | 32.910 | 48.309 | -31.88% |
| TP1 64/4 | 16 | 2048 | 32.170 | 48.544 | -33.73% |
| TP4 16/1 | 32 | 128 | 23.978 | 19.917 | +20.39% |
| TP4 16/1 | 32 | 512 | 28.054 | 26.016 | +7.84% |
| TP4 16/1 | 32 | 1024 | 32.170 | 26.022 | +23.62% |
| TP4 16/1 | 32 | 2048 | 32.170 | 32.173 | -0.01% |
| TP4 16/1 | 64 | 128 | 26.022 | 19.882 | +30.89% |
| TP4 16/1 | 64 | 512 | 28.080 | 36.278 | -22.60% |
| TP4 16/1 | 64 | 1024 | 32.182 | 36.272 | -11.27% |
| TP4 16/1 | 64 | 2048 | 32.168 | 48.550 | -33.74% |

The largest backend run CV was 2.274% (CUTLASS, TP1 batch 16, length 1024);
all other backend run CVs were below 0.55%. Applying
`Delta > max(5%, 2 CV)`, every claimed win or loss in the table is stable;
the near-zero rows are ties.

## Dispatch implication

The geometry-only proposal is incomplete. At its previously selected batch
cutoffs, CUTLASS reverses from a stable loss at 128 tokens to a stable win from
512 tokens onward:

- TP1 64/4: batch at least 16 **and** effective KV length at least 512;
- TP4 16/1: batch at least 64 **and** effective KV length at least 512.

These are conservative rules for the uniformly sized, page-aligned cases that
were measured. TP1 batch 8 has a narrow length-specific CUTLASS win at 512 but
ties at 1024 and 2048, so it should not motivate a non-monotonic special case.
TP4 batch 32 never has a stable CUTLASS win.

The experiment does not yet define a production key for a heterogeneous batch
whose requests have different sequence lengths. Updating the project policy or
upstream patch requires choosing a graph-stable aggregate (for example total
valid sparse blocks) and validating mixed-length batches. Until then, the
existing geometry-only policy and patch must not be treated as production-safe.

## Fusion implication

This result changes backend dispatch, not the fusion go/no-go. It supplies no
new evidence that the Triton partial/merge pair should be fused. The existing
fusion no-go remains unchanged.
