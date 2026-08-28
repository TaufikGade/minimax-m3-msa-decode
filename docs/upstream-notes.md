# Upstream notes

## Pinned source

- Project: vLLM
- Commit: d4da0c5
- Original path: vllm/models/minimax_m3/
- Local snapshot: vendor/vllm_msa_ref/
- License: Apache-2.0

The snapshot was supplied with Assignment 02 so experiments do not change when
upstream vLLM moves. Do not edit files under vendor/. Project-owned variants
belong under kernels/.

## Relevant upstream behavior

- Decode defaults to Triton split-K followed by a separate LSE merge kernel.
- The opt-in CUTLASS decode path targets SM100.
- The pinned dispatch threshold uses CUTLASS from batch size 16 for TP1 and TP4.
- Fixed shape: top-k 16, page size 128, head dimension 128, up to 64 query
  heads and 4 KV heads.

The threshold is an observation to reproduce, not evidence by itself.
