# SPDX-License-Identifier: Apache-2.0
"""Summarize run-to-run medians around the CUTLASS dispatch boundary."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups: dict[tuple[int, int, int, int, str], list[float]] = defaultdict(list)
    paired: dict[
        tuple[Path, int, int, int, int], dict[str, float]
    ] = defaultdict(dict)

    for path in args.inputs:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["component"] != "full" or row["execution"] != "graph":
                    continue
                batch = int(row["batch"])
                effective_kv_len = int(row.get("effective_kv_len", 2048))
                num_heads = int(row["num_heads"])
                num_kv_heads = int(row["num_kv_heads"])
                backend = row["backend"]
                median = float(row["median_us"])
                shape = (effective_kv_len, batch, num_heads, num_kv_heads)
                groups[(*shape, backend)].append(median)
                paired[(path, *shape)][backend] = median

    rows: list[dict[str, object]] = []
    shape_keys = sorted({key[:4] for key in groups})
    for effective_kv_len, batch, num_heads, num_kv_heads in shape_keys:
        cutlass = groups[
            (effective_kv_len, batch, num_heads, num_kv_heads, "cutlass")
        ]
        triton = groups[
            (effective_kv_len, batch, num_heads, num_kv_heads, "triton")
        ]
        deltas = []
        for (path, kv_len, b, h, hk), values in paired.items():
            if (kv_len, b, h, hk) == (
                effective_kv_len,
                batch,
                num_heads,
                num_kv_heads,
            ):
                deltas.append((values["cutlass"] / values["triton"] - 1.0) * 100.0)
        for backend, samples in (("cutlass", cutlass), ("triton", triton)):
            mean = statistics.mean(samples)
            stdev = statistics.pstdev(samples)
            rows.append(
                {
                    "effective_kv_len": effective_kv_len,
                    "batch": batch,
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "backend": backend,
                    "runs": len(samples),
                    "mean_of_medians_us": mean,
                    "stdev_of_medians_us": stdev,
                    "run_cv_pct": stdev / mean * 100.0,
                    "min_median_us": min(samples),
                    "max_median_us": max(samples),
                    "paired_cutlass_vs_triton_mean_pct": statistics.mean(deltas),
                    "paired_cutlass_vs_triton_stdev_pct": statistics.pstdev(deltas),
                }
            )

    if not rows:
        raise RuntimeError("no graph/full rows found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
