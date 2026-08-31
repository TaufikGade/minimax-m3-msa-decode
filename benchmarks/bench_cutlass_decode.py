# SPDX-License-Identifier: Apache-2.0
"""CUTLASS/Triton sparse-decode crossover benchmark."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from pathlib import Path
from typing import Callable

import torch

from cutlass_harness import (
    launch_cutlass_attention,
    launch_cutlass_full,
    launch_cutlass_metadata,
    make_cutlass_case,
)
from msa_harness import launch_triton_baseline, reference_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument(
        "--effective-kv-len",
        type=int,
        default=2048,
        help="Runtime KV length; allocation/top-k capacity remains 2048 tokens",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
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


def measure_eager(
    function: Callable[[], object], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        function()
        end.record()
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def measure_graph(
    function: Callable[[], object], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    graph = capture(function)
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


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"], text=True
    ).strip()


def accuracy(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    return max_abs, cosine


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.num_heads % args.num_kv_heads:
        raise ValueError("num-heads must be divisible by num-kv-heads")
    if min(args.batches) < 1 or args.warmup < 1 or args.iterations < 1:
        raise ValueError("batches, warmup, and iterations must be positive")
    if not 1 <= args.effective_kv_len <= 2048:
        raise ValueError("effective-kv-len must be in [1, 2048]")

    rows: list[dict[str, object]] = []
    commit = git_commit()
    for batch in args.batches:
        case = make_cutlass_case(
            batch,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            effective_kv_len=args.effective_kv_len,
            seed=args.seed,
        )
        expected = reference_decode(case.baseline)
        triton_out = launch_triton_baseline(
            case.baseline, case.baseline_workspace
        ).clone()
        cutlass_out = launch_cutlass_full(case).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(triton_out, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cutlass_out, expected, rtol=5e-2, atol=5e-2)
        triton_error, triton_cosine = accuracy(triton_out, expected)
        cutlass_error, cutlass_cosine = accuracy(cutlass_out, expected)

        functions: list[tuple[str, str, Callable[[], object], float, float]] = [
            (
                "cutlass",
                "metadata",
                lambda: launch_cutlass_metadata(case),
                cutlass_error,
                cutlass_cosine,
            ),
            (
                "cutlass",
                "attention",
                lambda: launch_cutlass_attention(case),
                cutlass_error,
                cutlass_cosine,
            ),
            (
                "cutlass",
                "full",
                lambda: launch_cutlass_full(case),
                cutlass_error,
                cutlass_cosine,
            ),
            (
                "triton",
                "full",
                lambda: launch_triton_baseline(
                    case.baseline, case.baseline_workspace
                ),
                triton_error,
                triton_cosine,
            ),
        ]

        # Compile all paths before any timed or captured region.
        for _, _, function, _, _ in functions:
            function()
        torch.cuda.synchronize()

        for execution, measure in (("eager", measure_eager), ("graph", measure_graph)):
            for backend, component, function, max_abs, cosine in functions:
                samples = measure(function, args.warmup, args.iterations)
                row = {
                    "commit": commit,
                    "gpu": torch.cuda.get_device_name(),
                    "batch": batch,
                    "num_heads": args.num_heads,
                    "num_kv_heads": args.num_kv_heads,
                    "effective_kv_len": args.effective_kv_len,
                    "backend": backend,
                    "component": component,
                    "execution": execution,
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
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
