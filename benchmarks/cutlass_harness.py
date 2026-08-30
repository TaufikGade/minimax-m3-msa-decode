# SPDX-License-Identifier: Apache-2.0
"""Standalone CUTLASS SM100 sparse-decode adapter for benchmark use."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from msa_harness import DecodeCase, DecodeWorkspace, allocate_workspace, make_case


@triton.jit
def _update_runtime_metadata_kernel(
    seq_lens_ptr,
    kv_segment_lens_ptr,
    qo_offset_ptr,
    num_rows: tl.constexpr,
    decode_query_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    request = offsets // decode_query_len
    local_query = offsets % decode_query_len
    seq_len = tl.load(seq_lens_ptr + request, mask=mask)
    tl.store(kv_segment_lens_ptr + offsets, seq_len, mask=mask)
    tl.store(
        qo_offset_ptr + offsets,
        seq_len - decode_query_len + local_query,
        mask=mask,
    )


@dataclass
class CutlassDecodeCase:
    baseline: DecodeCase
    baseline_workspace: DecodeWorkspace
    query_fp8: torch.Tensor
    query_scale: float
    key_scale: float
    value_scale: float
    topk: torch.Tensor
    output: torch.Tensor
    plan: object
    page_table: torch.Tensor


def _require_scalar_scale(case: DecodeCase, name: str) -> float:
    value = getattr(case, name)
    if value is None or value.numel() != 1:
        raise ValueError("CUTLASS decode smoke requires scalar FP8 KV scales")
    return float(value.item())


@torch.inference_mode()
def make_cutlass_case(
    batch: int,
    *,
    num_heads: int,
    num_kv_heads: int,
    seed: int = 20260829,
) -> CutlassDecodeCase:
    """Create one input shared by CUTLASS and the pinned Triton baseline."""
    from fmha_sm100.api import fmha_sm100_plan

    case = make_case(
        batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        fp8=True,
        scale_mode="scalar",
        seed=seed,
    )

    # CUTLASS requires strictly ascending logical block indices. Attention is
    # permutation invariant, so use this same order for the Triton comparison.
    case.topk_idx = torch.sort(case.topk_idx, dim=-1).values.contiguous()
    cutlass_topk = case.topk_idx.permute(1, 0, 2).contiguous()

    # Quantize Q independently and pass its dequantization scale to CUTLASS.
    # Keep the original BF16 Q in ``case`` for Triton and the FP32 reference.
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    query_scale = max(float(case.q.float().abs().max().item()) / fp8_max, 1e-8)
    query_fp8 = (case.q.float() / query_scale).to(torch.float8_e4m3fn)

    qo_lens_cpu = torch.full((batch,), case.decode_query_len, dtype=torch.int32)
    kv_lens_cpu = case.seq_lens.cpu()
    plan = fmha_sm100_plan(
        qo_lens_cpu,
        kv_lens_cpu,
        num_heads,
        num_kv_heads=num_kv_heads,
        qo_offset=kv_lens_cpu - qo_lens_cpu,
        page_size=case.kv_cache.shape[2],
        output_maxscore=False,
        kv_block_num=cutlass_topk.shape[-1],
        causal=True,
        sparse_kernel_mode="decode",
        use_fp8_kvcache=True,
        split_prefill_decode=False,
        device=case.q.device,
    )

    plan_info = plan[3]
    page_table_stride = int(case.block_table.stride(0))
    row_starts = (
        torch.arange(batch, dtype=torch.int32, device=case.q.device)
        .mul_(page_table_stride)
        .repeat_interleave(case.decode_query_len)
    )
    page_indptr = torch.cat(
        (
            row_starts,
            torch.tensor(
                [batch * page_table_stride],
                dtype=torch.int32,
                device=case.q.device,
            ),
        )
    )
    plan_info["kv_page_indptr"].copy_(page_indptr)

    return CutlassDecodeCase(
        baseline=case,
        baseline_workspace=allocate_workspace(case),
        query_fp8=query_fp8,
        query_scale=query_scale,
        key_scale=_require_scalar_scale(case, "k_scale"),
        value_scale=_require_scalar_scale(case, "v_scale"),
        topk=cutlass_topk,
        output=torch.empty_like(case.q),
        plan=plan,
        page_table=case.block_table.view(-1),
    )


@torch.inference_mode()
def launch_cutlass_metadata(case: CutlassDecodeCase) -> None:
    plan_info = case.plan[3]
    num_rows = case.baseline.q.shape[0]
    _update_runtime_metadata_kernel[(triton.cdiv(num_rows, 128),)](
        case.baseline.seq_lens,
        plan_info["kv_segment_lens"],
        plan_info["qo_offset"],
        num_rows=num_rows,
        decode_query_len=case.baseline.decode_query_len,
        BLOCK_SIZE=128,
    )


@torch.inference_mode()
def launch_cutlass_attention(case: CutlassDecodeCase) -> torch.Tensor:
    from fmha_sm100.api import fmha_sm100

    key, value = case.baseline.kv_cache.split(128, dim=-1)
    fmha_sm100(
        case.query_fp8,
        key,
        value,
        case.plan,
        kv_indices=case.page_table,
        kv_block_indexes=case.topk,
        out=case.output,
        output_maxscore=False,
        output_o=True,
        sm_scale=case.baseline.sm_scale,
        q_scale=case.query_scale,
        k_scale=case.key_scale,
        v_scale=case.value_scale,
        o_scale=1.0,
    )
    return case.output


@torch.inference_mode()
def launch_cutlass_full(case: CutlassDecodeCase) -> torch.Tensor:
    launch_cutlass_metadata(case)
    return launch_cutlass_attention(case)
