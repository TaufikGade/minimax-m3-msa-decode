# SPDX-License-Identifier: Apache-2.0
"""Paired CUDA Graph timing for partial-only, merge-only, and full decode."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Callable

import torch

from msa_harness import (
    allocate_workspace,
    default_num_chunks,
    launch_triton_baseline,
    launch_triton_merge,
    launch_triton_partial,
    make_case,
    reference_decode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--order",
        nargs=3,
        choices=("partial", "merge", "full"),
        default=("partial", "full", "merge"),
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def capture(function: Callable[[], object]) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph


def measure(
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
    if args.batch < 1 or args.warmup < 1 or args.iterations < 1:
        raise ValueError("batch, warmup, and iterations must be positive")
    if len(set(args.order)) != 3:
        raise ValueError("order must contain partial, merge, and full exactly once")

    case = make_case(args.batch, fp8=True, seed=args.seed)
    chunks = args.chunks
    if chunks is None:
        chunks = default_num_chunks(case)
    workspace = allocate_workspace(case, chunks)

    expected = reference_decode(case)
    actual = launch_triton_baseline(case, workspace).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()

    functions: dict[str, Callable[[], object]] = {
        "partial": lambda: launch_triton_partial(case, workspace),
        "merge": lambda: launch_triton_merge(case, workspace),
        "full": lambda: launch_triton_baseline(case, workspace),
    }

    # Compile every path and leave valid partial/LSE data for merge-only replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name in ("partial", "merge", "full"):
            functions[name]()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graphs = {name: capture(function) for name, function in functions.items()}
    rows = []
    for mode in args.order:
        samples = measure(graphs[mode], args.warmup, args.iterations)
        row = {
            "gpu": torch.cuda.get_device_name(),
            "batch": args.batch,
            "dtype": str(case.kv_cache.dtype),
            "num_chunks": workspace.num_chunks,
            "mode": mode,
            "median_us": statistics.median(samples),
            "p95_us": percentile(samples, 0.95),
            "stdev_us": statistics.pstdev(samples),
            "iterations": len(samples),
            "max_abs_error": max_abs,
            "cosine_similarity": cosine,
            "seed": args.seed,
        }
        rows.append(row)
        print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream_out:
        writer = csv.DictWriter(stream_out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
