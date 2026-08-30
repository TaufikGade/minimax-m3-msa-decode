# SPDX-License-Identifier: Apache-2.0
"""Summarize paired full-versus-partial CUDA Graph measurements."""

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
    trials: dict[int, list[dict[str, float]]] = defaultdict(list)
    metadata: dict[int, dict[str, str]] = {}

    for path in args.inputs:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        by_mode = {row["mode"]: row for row in rows}
        if set(by_mode) != {"partial", "merge", "full"}:
            raise ValueError(f"{path}: expected partial, merge, and full rows")
        batch = int(rows[0]["batch"])
        # Older TP1 result files predate explicit head-count columns. Preserve
        # their reproducibility by assigning the original 64/4 defaults.
        for row in rows:
            row.setdefault("num_heads", "64")
            row.setdefault("num_kv_heads", "4")
            row.setdefault(
                "scale_mode",
                "scalar" if "float8" in row["dtype"] else "none",
            )
        stable = (
            "gpu",
            "batch",
            "num_heads",
            "num_kv_heads",
            "dtype",
            "scale_mode",
            "num_chunks",
            "iterations",
            "seed",
        )
        for field in stable:
            values = {row[field] for row in rows}
            if len(values) != 1:
                raise ValueError(f"{path}: inconsistent {field}: {values}")
        current_metadata = {field: rows[0][field] for field in stable}
        if batch in metadata and metadata[batch] != current_metadata:
            raise ValueError(f"batch {batch}: metadata changed across trials")
        metadata[batch] = current_metadata

        partial = float(by_mode["partial"]["median_us"])
        merge = float(by_mode["merge"]["median_us"])
        full = float(by_mode["full"]["median_us"])
        incremental = full - partial
        trials[batch].append(
            {
                "partial_us": partial,
                "merge_us": merge,
                "full_us": full,
                "incremental_merge_us": incremental,
                "eliminate_merge_upper_bound_pct": 100.0 * incremental / full,
            }
        )

    summaries = []
    for batch, values in sorted(trials.items()):
        if len(values) < 2:
            raise ValueError(f"batch {batch}: at least two paired trials required")

        def mean(field: str) -> float:
            return statistics.mean(value[field] for value in values)

        bounds = [value["eliminate_merge_upper_bound_pct"] for value in values]
        increments = [value["incremental_merge_us"] for value in values]
        meta = metadata[batch]
        summaries.append(
            {
                "gpu": meta["gpu"],
                "batch": batch,
                "num_heads": int(meta["num_heads"]),
                "num_kv_heads": int(meta["num_kv_heads"]),
                "dtype": meta["dtype"],
                "scale_mode": meta["scale_mode"],
                "num_chunks": int(meta["num_chunks"]),
                "runs": len(values),
                "iterations_per_mode_per_run": int(meta["iterations"]),
                "mean_partial_us": mean("partial_us"),
                "mean_merge_only_us": mean("merge_us"),
                "mean_full_us": mean("full_us"),
                "mean_incremental_merge_us": statistics.mean(increments),
                "sample_std_incremental_merge_us": statistics.stdev(increments),
                "mean_eliminate_merge_upper_bound_pct": statistics.mean(bounds),
                "sample_std_upper_bound_pct": statistics.stdev(bounds),
                "min_upper_bound_pct": min(bounds),
                "max_upper_bound_pct": max(bounds),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
