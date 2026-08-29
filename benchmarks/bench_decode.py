# SPDX-License-Identifier: Apache-2.0
"""Benchmark the preallocated pinned Triton decode baseline."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch

from msa_harness import allocate_workspace, launch_triton_baseline, make_case


def measure_eager(case, workspace, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        launch_triton_baseline(case, workspace)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_triton_baseline(case, workspace)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


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


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for batch in args.batches:
        case = make_case(batch, fp8=not args.bf16)
        workspace = allocate_workspace(case)
        modes = ["eager", "graph"] if args.mode == "both" else [args.mode]
        for mode in modes:
            function = measure_eager if mode == "eager" else measure_graph
            samples = function(case, workspace, args.warmup, args.iterations)
            row = {
                "gpu": torch.cuda.get_device_name(),
                "batch": batch,
                "dtype": str(case.kv_cache.dtype),
                "mode": mode,
                "num_chunks": workspace.num_chunks,
                "median_us": statistics.median(samples),
                "p95_us": percentile(samples, 0.95),
                "stdev_us": statistics.pstdev(samples),
                "iterations": len(samples),
            }
            rows.append(row)
            print(row)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
