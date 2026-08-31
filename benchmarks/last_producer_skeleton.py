# SPDX-License-Identifier: Apache-2.0
"""Synthetic last-producer resource and CUDA Graph correctness skeleton.

This intentionally contains no attention computation. It preserves the
proposed partial/LSE state shapes and completion protocol so compiler resource
use and cross-CTA visibility can be measured before a functional prototype.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


SCALE_SCALAR = 1
SCALE_PER_TOKEN = 2
BLOCK_H = 16
BLOCK_D = 128


@triton.jit
def _last_producer_skeleton_kernel(
    input_o_ptr,
    input_lse_ptr,
    scale_ptr,
    workspace_o_ptr,
    workspace_lse_ptr,
    counter_ptr,
    output_ptr,
    NUM_CHUNKS: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    ENABLE_LAST_PRODUCER: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid // NUM_CHUNKS
    chunk = pid % NUM_CHUNKS
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    input_base = (group * NUM_CHUNKS + chunk) * BLOCK_SIZE_H * BLOCK_SIZE_D
    input_ptrs = input_o_ptr + input_base + off_h[:, None] * BLOCK_SIZE_D + off_d[None, :]
    partial = tl.load(input_ptrs).to(tl.float32)
    if SCALE_MODE == 1:
        scale = tl.load(scale_ptr)
    else:
        scale_base = (group * NUM_CHUNKS + chunk) * BLOCK_SIZE_H
        scale = tl.load(scale_ptr + scale_base + off_h)
    partial *= scale if SCALE_MODE == 1 else scale[:, None]

    workspace_base = (chunk * tl.num_programs(0) // NUM_CHUNKS + group) * BLOCK_SIZE_H * BLOCK_SIZE_D
    workspace_ptrs = (
        workspace_o_ptr
        + workspace_base
        + off_h[:, None] * BLOCK_SIZE_D
        + off_d[None, :]
    )
    tl.store(workspace_ptrs, partial.to(workspace_o_ptr.dtype.element_ty))

    lse_base = (group * NUM_CHUNKS + chunk) * BLOCK_SIZE_H
    lse = tl.load(input_lse_ptr + lse_base + off_h)
    workspace_lse_base = (chunk * tl.num_programs(0) // NUM_CHUNKS + group) * BLOCK_SIZE_H
    tl.store(workspace_lse_ptr + workspace_lse_base + off_h, lse)

    if ENABLE_LAST_PRODUCER:
        # All lanes must finish their partial/LSE stores before the designated
        # program-level completion atomic publishes this producer.
        tl.debug_barrier()
        old = tl.atomic_add(
            counter_ptr + group,
            1,
            sem="acq_rel",
            scope="gpu",
        )

        if old == NUM_CHUNKS - 1:
            merged = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), tl.float32)
            normalizer = tl.zeros((BLOCK_SIZE_H,), tl.float32)
            lse_max = tl.full((BLOCK_SIZE_H,), float("-inf"), tl.float32)
            for reduce_chunk in tl.static_range(0, NUM_CHUNKS):
                reduce_base = (
                    reduce_chunk * tl.num_programs(0) // NUM_CHUNKS + group
                ) * BLOCK_SIZE_H * BLOCK_SIZE_D
                value = tl.load(
                    workspace_o_ptr
                    + reduce_base
                    + off_h[:, None] * BLOCK_SIZE_D
                    + off_d[None, :]
                ).to(tl.float32)
                reduce_lse_base = (
                    reduce_chunk * tl.num_programs(0) // NUM_CHUNKS + group
                ) * BLOCK_SIZE_H
                chunk_lse = tl.load(workspace_lse_ptr + reduce_lse_base + off_h)
                next_max = tl.maximum(lse_max, chunk_lse)
                old_weight = tl.exp2(lse_max - next_max)
                new_weight = tl.exp2(chunk_lse - next_max)
                merged = merged * old_weight[:, None] + value * new_weight[:, None]
                normalizer = normalizer * old_weight + new_weight
                lse_max = next_max

            merged /= normalizer[:, None]
            output_base = group * BLOCK_SIZE_H * BLOCK_SIZE_D
            tl.store(
                output_ptr
                + output_base
                + off_h[:, None] * BLOCK_SIZE_D
                + off_d[None, :],
                merged.to(output_ptr.dtype.element_ty),
            )
            tl.atomic_xchg(
                counter_ptr + group,
                0,
                sem="release",
                scope="gpu",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument(
        "--scale-mode", choices=("scalar", "per_token"), required=True
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def measure_graph(
    graph: torch.cuda.CUDAGraph, warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.groups < 1 or args.chunks not in (1, 2, 4, 8, 16):
        raise ValueError("groups must be positive and chunks a power of two <= 16")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape_o = (args.groups, args.chunks, BLOCK_H, BLOCK_D)
    shape_lse = (args.groups, args.chunks, BLOCK_H)
    input_o = torch.randn(
        shape_o, dtype=torch.bfloat16, device=device, generator=generator
    )
    input_lse = torch.randn(
        shape_lse, dtype=torch.float32, device=device, generator=generator
    )
    scale_mode = SCALE_SCALAR if args.scale_mode == "scalar" else SCALE_PER_TOKEN
    if scale_mode == SCALE_SCALAR:
        scale = torch.tensor(0.75, dtype=torch.float32, device=device)
        scaled = (input_o.float() * scale).to(torch.bfloat16).float()
    else:
        scale = 0.5 + torch.rand(
            shape_lse, dtype=torch.float32, device=device, generator=generator
        )
        scaled = (input_o.float() * scale[..., None]).to(torch.bfloat16).float()

    # Workspace is chunk-major, matching the real split-K layout.
    workspace_o = torch.empty(
        (args.chunks, args.groups, BLOCK_H, BLOCK_D),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace_lse = torch.empty(
        (args.chunks, args.groups, BLOCK_H),
        dtype=torch.float32,
        device=device,
    )
    counters = torch.zeros(args.groups, dtype=torch.int32, device=device)
    output = torch.empty(
        (args.groups, BLOCK_H, BLOCK_D), dtype=torch.bfloat16, device=device
    )

    weights = torch.softmax(input_lse * math.log(2.0), dim=1)
    expected = (scaled * weights[..., None]).sum(dim=1).to(torch.bfloat16)
    grid = (args.groups * args.chunks,)
    launch_args = (
        input_o,
        input_lse,
        scale,
        workspace_o,
        workspace_lse,
        counters,
        output,
    )
    launch_kwargs = {
        "NUM_CHUNKS": args.chunks,
        "SCALE_MODE": scale_mode,
        "ENABLE_LAST_PRODUCER": True,
        "BLOCK_SIZE_H": BLOCK_H,
        "BLOCK_SIZE_D": BLOCK_D,
        "num_warps": 8,
    }

    compiled = _last_producer_skeleton_kernel.warmup(
        *launch_args, grid=grid, **launch_kwargs
    )
    compiled._init_handles()
    producer_kwargs = {**launch_kwargs, "ENABLE_LAST_PRODUCER": False}
    producer_compiled = _last_producer_skeleton_kernel.warmup(
        *launch_args, grid=grid, **producer_kwargs
    )
    producer_compiled._init_handles()
    _last_producer_skeleton_kernel[grid](*launch_args, **launch_kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    if torch.count_nonzero(counters).item() != 0:
        raise AssertionError("completion counters were not reset after eager launch")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _last_producer_skeleton_kernel[grid](*launch_args, **launch_kwargs)
    producer_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(producer_graph):
        _last_producer_skeleton_kernel[grid](*launch_args, **producer_kwargs)
    for _ in range(args.graph_replays):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    if torch.count_nonzero(counters).item() != 0:
        raise AssertionError("completion counters were not reset after Graph replay")

    producer_samples = measure_graph(producer_graph, args.warmup, args.iterations)
    samples = measure_graph(graph, args.warmup, args.iterations)
    producer_median = statistics.median(producer_samples)
    full_median = statistics.median(samples)

    ptx = compiled.asm.get("ptx", "")
    atomic_lines = [
        line.strip()
        for line in ptx.splitlines()
        if "atom." in line or "red." in line or "acquire" in line or "release" in line
    ]
    max_abs = (output.float() - expected.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    result = {
        "gpu": torch.cuda.get_device_name(),
        "triton": triton.__version__,
        "groups": args.groups,
        "chunks": args.chunks,
        "scale_mode": args.scale_mode,
        "producer_ctas": args.groups * args.chunks,
        "last_reducers": args.groups,
        "registers_per_thread": compiled.n_regs,
        "local_spill_words_per_thread": compiled.n_spills,
        "dynamic_smem_bytes": compiled.metadata.shared,
        "producer_registers_per_thread": producer_compiled.n_regs,
        "producer_local_spill_words_per_thread": producer_compiled.n_spills,
        "producer_dynamic_smem_bytes": producer_compiled.metadata.shared,
        "num_warps": compiled.metadata.num_warps,
        "graph_replays_checked": args.graph_replays,
        "producer_graph_median_us": producer_median,
        "producer_graph_p95_us": percentile(producer_samples, 0.95),
        "graph_median_us": full_median,
        "graph_p95_us": percentile(samples, 0.95),
        "graph_stdev_us": statistics.pstdev(samples),
        "last_producer_increment_us": full_median - producer_median,
        "last_producer_increment_pct_of_full": (
            (full_median - producer_median) / full_median * 100.0
        ),
        "max_abs_error": max_abs,
        "cosine_similarity": cosine,
        "counter_nonzero": int(torch.count_nonzero(counters).item()),
        "ptx_atomic_lines": atomic_lines,
        "seed": args.seed,
    }
    print(json.dumps(result, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
