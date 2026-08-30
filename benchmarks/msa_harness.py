# SPDX-License-Identifier: Apache-2.0
"""Standalone inputs, reference, and preallocated Triton baseline launcher."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

PAGE_SIZE = 128
HEAD_DIM = 128
TOPK = 16


def load_vendor_sparse_attn():
    """Load the pinned kernel without installing the full vLLM package."""
    module_name = "_vendored_minimax_m3_sparse_attn"
    if module_name in sys.modules:
        return sys.modules[module_name]

    class _Platform:
        @staticmethod
        def is_arch_support_pdl() -> bool:
            return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

    vllm = types.ModuleType("vllm")
    platforms = types.ModuleType("vllm.platforms")
    triton_utils = types.ModuleType("vllm.triton_utils")
    platforms.current_platform = _Platform()

    import triton
    import triton.language as tl

    triton_utils.triton = triton
    triton_utils.tl = tl
    sys.modules.setdefault("vllm", vllm)
    sys.modules["vllm.platforms"] = platforms
    sys.modules["vllm.triton_utils"] = triton_utils

    source = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "vllm_msa_ref"
        / "sparse_attn.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load vendored kernel from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class DecodeCase:
    q: torch.Tensor
    kv_cache: torch.Tensor
    topk_idx: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    k_scale: torch.Tensor | None
    v_scale: torch.Tensor | None
    num_kv_heads: int
    decode_query_len: int = 1

    @property
    def sm_scale(self) -> float:
        return self.q.shape[-1] ** -0.5

    @property
    def scale_mode(self) -> str:
        if self.k_scale is None:
            return "none"
        if self.k_scale.numel() == 1:
            return "scalar"
        return "per_token_head"


@dataclass
class DecodeWorkspace:
    output: torch.Tensor
    partial: torch.Tensor
    lse: torch.Tensor
    num_chunks: int


def make_case(
    batch: int,
    *,
    num_heads: int = 64,
    num_kv_heads: int = 4,
    seq_len: int = TOPK * PAGE_SIZE,
    fp8: bool = True,
    scale_mode: str = "scalar",
    seed: int = 20260829,
    device: str = "cuda",
) -> DecodeCase:
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if seq_len > TOPK * PAGE_SIZE:
        raise ValueError("minimal harness supports sequence lengths up to 2048")
    if scale_mode not in ("scalar", "per_token_head"):
        raise ValueError("scale_mode must be scalar or per_token_head")

    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch,
        num_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.1

    logical_blocks = math.ceil(seq_len / PAGE_SIZE)
    physical_pages = batch * logical_blocks
    kv_bf16 = torch.randn(
        physical_pages,
        num_kv_heads,
        PAGE_SIZE,
        2 * HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    block_table = torch.arange(
        physical_pages, dtype=torch.int32, device=device
    ).reshape(batch, logical_blocks)
    for request in range(batch):
        permutation = torch.randperm(
            logical_blocks, generator=generator, device=device
        )
        block_table[request] = block_table[request, permutation]

    topk_idx = torch.empty(
        num_kv_heads, batch, TOPK, dtype=torch.int32, device=device
    )
    base = torch.arange(TOPK, dtype=torch.int32, device=device)
    for kv_head in range(num_kv_heads):
        for request in range(batch):
            topk_idx[kv_head, request] = base[
                torch.randperm(TOPK, generator=generator, device=device)
            ]

    k_scale = v_scale = None
    kv_cache = kv_bf16
    if fp8:
        if scale_mode == "scalar":
            k_scale = torch.tensor(0.25, dtype=torch.float32, device=device)
            v_scale = torch.tensor(0.50, dtype=torch.float32, device=device)
            k_quant_scale = k_scale
            v_quant_scale = v_scale
        else:
            # Match the upstream [KV head, flattened physical token] layout.
            # Vary the values so correctness checks also exercise physical-page
            # indexing instead of only validating the tensor shape.
            physical_tokens = physical_pages * PAGE_SIZE
            k_scale = 0.20 + 0.10 * torch.rand(
                num_kv_heads,
                physical_tokens,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            v_scale = 0.40 + 0.20 * torch.rand(
                num_kv_heads,
                physical_tokens,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            k_quant_scale = (
                k_scale.reshape(num_kv_heads, physical_pages, PAGE_SIZE)
                .permute(1, 0, 2)
                .unsqueeze(-1)
            )
            v_quant_scale = (
                v_scale.reshape(num_kv_heads, physical_pages, PAGE_SIZE)
                .permute(1, 0, 2)
                .unsqueeze(-1)
            )
        kv_cache = torch.empty_like(kv_bf16, dtype=torch.float8_e4m3fn)
        kv_cache[..., :HEAD_DIM] = (
            kv_bf16[..., :HEAD_DIM].float() / k_quant_scale
        ).to(kv_cache.dtype)
        kv_cache[..., HEAD_DIM:] = (
            kv_bf16[..., HEAD_DIM:].float() / v_quant_scale
        ).to(kv_cache.dtype)

    return DecodeCase(
        q=q,
        kv_cache=kv_cache,
        topk_idx=topk_idx,
        block_table=block_table,
        seq_lens=torch.full(
            (batch,), seq_len, dtype=torch.int32, device=device
        ),
        k_scale=k_scale,
        v_scale=v_scale,
        num_kv_heads=num_kv_heads,
    )


def default_num_chunks(case: DecodeCase) -> int:
    """Return the chunk heuristic used by the pinned upstream wrapper."""
    total_q, num_heads, head_dim = case.q.shape
    max_topk = case.topk_idx.shape[-1]
    target = max(
        1,
        min(max_topk, 256 // max(1, total_q * case.num_kv_heads)),
    )
    return 1 << (target.bit_length() - 1)


def allocate_workspace(
    case: DecodeCase, num_chunks: int | None = None
) -> DecodeWorkspace:
    total_q, num_heads, head_dim = case.q.shape
    max_topk = case.topk_idx.shape[-1]
    if num_chunks is None:
        num_chunks = default_num_chunks(case)
    if (
        num_chunks < 1
        or num_chunks > max_topk
        or num_chunks & (num_chunks - 1)
    ):
        raise ValueError(
            f"num_chunks must be a power of two in [1, {max_topk}]"
        )
    return DecodeWorkspace(
        output=torch.empty_like(case.q),
        partial=torch.empty(
            num_chunks,
            total_q,
            num_heads,
            head_dim,
            dtype=case.q.dtype,
            device=case.q.device,
        ),
        lse=torch.empty(
            num_chunks,
            total_q,
            num_heads,
            dtype=torch.float32,
            device=case.q.device,
        ),
        num_chunks=num_chunks,
    )


def _scale_launch_args(case: DecodeCase, workspace: DecodeWorkspace):
    op = load_vendor_sparse_attn()
    use_fp8 = case.kv_cache.dtype in op._FP8_DTYPES
    if use_fp8:
        scale_args = op._kv_scale_args(
            workspace.output,
            case.num_kv_heads,
            case.k_scale,
            case.v_scale,
        )
    else:
        scale_args = (
            workspace.output,
            workspace.output,
            0,
            0,
            0,
            0,
            op._KV_SCALE_NONE,
        )
    return op, use_fp8, scale_args


def _pdl_launch_options(use_pdl: bool) -> tuple[bool, dict[str, bool]]:
    pdl = use_pdl and torch.cuda.get_device_capability()[0] >= 9
    return pdl, {"launch_pdl": True} if pdl else {}


@torch.inference_mode()
def launch_triton_partial(
    case: DecodeCase,
    workspace: DecodeWorkspace,
    *,
    use_pdl: bool = True,
) -> None:
    """Launch only the pinned split-K partial kernel."""
    op, use_fp8, scale_args = _scale_launch_args(case, workspace)
    total_q, num_heads, head_dim = case.q.shape
    gqa_group_size = num_heads // case.num_kv_heads
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        scale_mode,
    ) = scale_args
    pdl, launch_options = _pdl_launch_options(use_pdl)

    partial = workspace.partial
    lse = workspace.lse
    grid = (total_q * workspace.num_chunks, case.num_kv_heads)
    op._gqa_sparse_decode_kernel[grid](
        case.q,
        case.kv_cache,
        k_scale_arg,
        v_scale_arg,
        case.topk_idx,
        partial,
        lse,
        case.block_table,
        case.seq_lens,
        total_q,
        gqa_group_size,
        head_dim,
        case.topk_idx.shape[-1],
        case.sm_scale,
        case.decode_query_len,
        *case.q.stride(),
        *case.kv_cache.stride(),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        *case.topk_idx.stride(),
        *partial.stride(),
        *lse.stride(),
        case.block_table.stride(0),
        BLOCK_SIZE_K=PAGE_SIZE,
        NUM_TOPK_CHUNKS=workspace.num_chunks,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=scale_mode,
        USE_PDL=pdl,
        **launch_options,
    )


@torch.inference_mode()
def launch_triton_merge(
    case: DecodeCase,
    workspace: DecodeWorkspace,
    *,
    use_pdl: bool = True,
) -> torch.Tensor:
    """Launch only the pinned LSE merge kernel."""
    op = load_vendor_sparse_attn()
    total_q, num_heads, head_dim = case.q.shape
    pdl, launch_options = _pdl_launch_options(use_pdl)
    partial = workspace.partial
    lse = workspace.lse
    op._merge_topk_attn_out_kernel[(total_q, num_heads)](
        partial,
        lse,
        workspace.output,
        head_dim,
        *partial.stride(),
        *lse.stride(),
        *workspace.output.stride(),
        NUM_TOPK_CHUNKS=workspace.num_chunks,
        USE_PDL=pdl,
        **launch_options,
    )
    return workspace.output


@torch.inference_mode()
def launch_triton_baseline(
    case: DecodeCase,
    workspace: DecodeWorkspace,
    *,
    use_pdl: bool = True,
) -> torch.Tensor:
    """Launch split-K and merge without allocations."""
    launch_triton_partial(case, workspace, use_pdl=use_pdl)
    return launch_triton_merge(case, workspace, use_pdl=use_pdl)


@torch.inference_mode()
def reference_decode(case: DecodeCase) -> torch.Tensor:
    """FP32 gathered-KV reference for decode_query_len=1."""
    if case.decode_query_len != 1:
        raise NotImplementedError("reference currently supports query length 1")
    batch, num_heads, head_dim = case.q.shape
    group = num_heads // case.num_kv_heads
    result = torch.empty_like(case.q)

    for request in range(batch):
        valid_tokens = int(case.seq_lens[request].item())
        valid_blocks = min(TOPK, math.ceil(valid_tokens / PAGE_SIZE))
        for kv_head in range(case.num_kv_heads):
            keys = []
            values = []
            for rank in range(valid_blocks):
                logical = int(case.topk_idx[kv_head, request, rank].item())
                physical = int(case.block_table[request, logical].item())
                block_tokens = min(PAGE_SIZE, valid_tokens - logical * PAGE_SIZE)
                if block_tokens <= 0:
                    continue
                page = case.kv_cache[physical, kv_head]
                key = page[:block_tokens, :head_dim].float()
                value = page[:block_tokens, head_dim:].float()
                if case.k_scale is not None:
                    if case.scale_mode == "scalar":
                        key = key * case.k_scale.float()
                        value = value * case.v_scale.float()
                    else:
                        token_start = physical * PAGE_SIZE
                        token_stop = token_start + block_tokens
                        key_scale = case.k_scale[
                            kv_head, token_start:token_stop
                        ].float()
                        value_scale = case.v_scale[
                            kv_head, token_start:token_stop
                        ].float()
                        key = key * key_scale[:, None]
                        value = value * value_scale[:, None]
                keys.append(key)
                values.append(value)
            key = torch.cat(keys, dim=0)
            value = torch.cat(values, dim=0)
            head_start = kv_head * group
            query = case.q[request, head_start : head_start + group].float()
            probability = torch.softmax(query @ key.T * case.sm_scale, dim=-1)
            result[request, head_start : head_start + group] = (
                probability @ value
            ).to(result.dtype)
    return result
