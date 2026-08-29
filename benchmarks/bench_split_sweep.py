# SPDX-License-Identifier: Apache-2.0
"""Measure split-K partial/merge time and sweep the number of chunks."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

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


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def measure_components(
    case, workspace, warmup: int, iterations: int
) -> tuple[list[float], list[float], list[float]]:
    for _ in range(warmup):
        launch_triton_baseline(case, workspace)
    torch.cuda.synchronize()
    partial_samples, merge_samples, total_samples = [], [], []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        middle = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_triton_partial(case, workspace)
        middle.record()
        launch_triton_merge(case, workspace)
        end.record()
        end.synchronize()
        partial_samples.append(start.elapsed_time(middle) * 1000.0)
        merge_samples.append(middle.elapsed_time(end) * 1000.0)
        total_samples.append(start.elapsed_time(end) * 1000.0)
    return partial_samples, merge_samples, total_samples


def measure_graph(case, workspace, warmup: int, iterations: int) -> list[float]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            launch_triton_baseline(case, workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_triton_baseline(case, workspace)
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def correctness(case, workspace, expected) -> tuple[float, float]:
    actual = launch_triton_baseline(case, workspace).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    return max_abs, cosine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--chunks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("results/raw/split-sweep.csv")
    )
    args = parser.parse_args()

    rows = []
    for batch in args.batches:
        case = make_case(batch, fp8=True)
        expected = reference_decode(case)
        upstream_chunks = default_num_chunks(case)
        for chunks in args.chunks:
            workspace = allocate_workspace(case, chunks)
            max_abs, cosine = correctness(case, workspace, expected)
            partial, merge, total = measure_components(
                case, workspace, args.warmup, args.iterations
            )
            graph = measure_graph(case, workspace, args.warmup, args.iterations)
            partial_median = statistics.median(partial)
            merge_median = statistics.median(merge)
            total_median = statistics.median(total)
            row = {
                "gpu": torch.cuda.get_device_name(),
                "batch": batch,
                "chunks": chunks,
                "is_upstream_default": chunks == upstream_chunks,
                "partial_median_us": partial_median,
                "merge_median_us": merge_median,
                "total_median_us": total_median,
                "graph_median_us": statistics.median(graph),
                "merge_fraction": merge_median / total_median,
                "total_p95_us": percentile(total, 0.95),
                "graph_p95_us": percentile(graph, 0.95),
                "total_stdev_us": statistics.pstdev(total),
                "graph_stdev_us": statistics.pstdev(graph),
                "max_abs_error": max_abs,
                "cosine": cosine,
                "iterations": args.iterations,
            }
            rows.append(row)
            print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
