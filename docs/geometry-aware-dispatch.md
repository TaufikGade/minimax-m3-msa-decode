# Geometry-aware decode dispatch

## Status

This is a project-owned policy proposal, not a modification to the pinned vLLM
snapshot and not a production integration. Its implementation is in
`kernels/cutlass_dispatch_policy.py`.

**Superseded by effective-length evidence:** the current implementation and
upstream patch encode geometry and batch only. The effective-KV-length sweep
shows that those rules select CUTLASS incorrectly at 128 tokens. Do not use the
implementation or patch as a production policy until a graph-stable length or
valid-block signal is added. See `docs/cutlass-kvlen-crossover-findings.md`.

## Evidence-bounded policy

For one-token decode with scalar FP8 KV scales on the measured NVIDIA B300:

| Head geometry | Interpretation | Triton | Measured CUTLASS region |
|---|---|---|---|
| 64 query / 4 KV | TP1-like | batch < 16 or effective KV < 512 | batch >= 16 and effective KV >= 512 |
| 16 query / 1 KV | TP4-like | batch < 64 or effective KV < 512 | batch >= 64 and effective KV >= 512 |

The batch thresholds come from ten independent Python processes per boundary
shape. A second 160-process sweep established that effective KV length changes
the decision: at 128 tokens CUTLASS was 9.06% slower for TP1 batch 16 and
30.89% slower for TP4 batch 64, while every tested length from 512 through 2048
was a stable CUTLASS win at those cutoffs. See
`docs/cutlass-crossover-findings.md`,
`docs/cutlass-kvlen-crossover-findings.md`, and the corresponding raw summaries.

The policy intentionally falls back to Triton for:

- per-token/head FP8 scales, because the measured CUTLASS harness and pinned
  wrapper accept scalar scales only;
- decode query lengths other than one;
- head geometries other than the exact measured 64/4 and 16/1 pairs.

Both measured geometries have GQA ratio 16, yet their thresholds differ by 4x.
The dispatch key therefore cannot be reduced to GQA ratio alone.

## Upstream integration

The pinned commit already represents per-token/head scaling with the distinct
`fp8_per_token_head` cache dtype. Its CUTLASS support guard accepts only `fp8`
and `fp8_e4m3`, so the metadata builder already has a static, graph-stable
signal and correctly rejects the per-token/head mode. No new runtime tensor
inspection or cache-key field is required.

This was checked directly at pinned vLLM commit `d4da0c5`:

- `vllm/config/cache.py` includes `fp8_per_token_head` in `CacheDType`;
- `vllm/utils/torch_utils.py` identifies per-token/head modes from the dtype;
- `vllm/model_executor/layers/quantization/kv_cache.py` keeps scalar host
  values at 1.0 for per-token/head caches because their scales are dynamic.

The minimal upstream change is to replace the common batch-16 cutoff with the
exact-geometry threshold table. The
[patch draft](../patches/vllm-d4da0c5-geometry-aware-cutlass-dispatch.patch)
records that proposal without modifying `vendor/`. The geometry-aware rule is
limited to one-token decode; speculative decode retains the pinned batch-16
behavior because this project did not measure its crossover. A
[companion test patch](../patches/vllm-d4da0c5-geometry-aware-cutlass-dispatch-tests.patch)
covers that distinction.

The source patch is validated against the snapshot with:

```bash
git apply --check --unidiff-zero -p5 \
  --directory=vendor/vllm_msa_ref \
  patches/vllm-d4da0c5-geometry-aware-cutlass-dispatch.patch
```

An upstream submission should add tests for TP1, TP4, per-token fallback,
E5M2 fallback, and unmeasured geometries.
