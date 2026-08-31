# Geometry-aware decode dispatch

## Status

This is a project-owned policy proposal, not a modification to the pinned vLLM
snapshot and not a production integration. Its implementation is in
`kernels/cutlass_dispatch_policy.py`.

## Evidence-bounded policy

For one-token decode with scalar FP8 KV scales on the measured NVIDIA B300:

| Head geometry | Interpretation | Triton | CUTLASS |
|---|---|---:|---:|
| 64 query / 4 KV | TP1-like | batch < 16 | batch >= 16 |
| 16 query / 1 KV | TP4-like | batch < 64 | batch >= 64 |

The thresholds come from ten independent Python processes per boundary shape.
At the selected boundaries CUTLASS was 28.91% faster for TP1 batch 16 and
28.94% faster for TP4 batch 64, with run CV below 0.26%. See
`docs/cutlass-crossover-findings.md` and
`results/raw/cutlass-boundary-noise-summary.csv`.

The policy intentionally falls back to Triton for:

- per-token/head FP8 scales, because the measured CUTLASS harness and pinned
  wrapper accept scalar scales only;
- decode query lengths other than one;
- head geometries other than the exact measured 64/4 and 16/1 pairs.

Both measured geometries have GQA ratio 16, yet their thresholds differ by 4x.
The dispatch key therefore cannot be reduced to GQA ratio alone.

## Integration gap

The pinned metadata builder selects and prepares a CUTLASS plan before the
runtime `k_scale` and `v_scale` tensors are inspected. A production upstream
integration needs an explicit, graph-stable scale-mode signal in the builder
or model configuration. Without that signal, automatically selecting CUTLASS
could silently route per-token/head scale cases through scalar scale arguments.

The next integration step is therefore an upstream design change, not a local
edit to `vendor/`: thread the configured KV scale granularity into metadata
planning, then replace the common batch-16 cutoff with the exact-geometry
policy and add upstream tests for TP1, TP4, per-token fallback, and unmeasured
geometries.
