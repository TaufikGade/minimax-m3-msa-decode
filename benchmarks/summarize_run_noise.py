# SPDX-License-Identifier: Apache-2.0
"""Summarize run-to-run variation from independent benchmark CSV files."""

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


def read_trials(paths: list[Path]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        graph_rows = [row for row in rows if row["mode"] == "graph"]
        batches = [int(row["batch"]) for row in graph_rows]
        if not graph_rows:
            raise ValueError(f"{path}: no graph rows")
        if len(batches) != len(set(batches)):
            raise ValueError(f"{path}: duplicate graph row for a batch")
        for row in graph_rows:
            row = dict(row)
            row["trial_file"] = str(path)
            grouped[int(row["batch"])].append(row)
    return grouped


def summarize(grouped: dict[int, list[dict[str, str]]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for batch, rows in sorted(grouped.items()):
        if len(rows) < 2:
            raise ValueError(f"batch {batch}: at least two independent runs required")

        stable_fields = ("gpu", "dtype", "num_chunks", "iterations")
        for field in stable_fields:
            values = {row[field] for row in rows}
            if len(values) != 1:
                raise ValueError(f"batch {batch}: inconsistent {field}: {values}")

        medians = [float(row["median_us"]) for row in rows]
        mean_median = statistics.mean(medians)
        sample_std = statistics.stdev(medians)
        cv = sample_std / mean_median
        summaries.append(
            {
                "gpu": rows[0]["gpu"],
                "batch": batch,
                "dtype": rows[0]["dtype"],
                "num_chunks": int(rows[0]["num_chunks"]),
                "runs": len(rows),
                "iterations_per_run": int(rows[0]["iterations"]),
                "mean_run_median_us": mean_median,
                "sample_std_run_median_us": sample_std,
                "cv_pct": 100.0 * cv,
                "significance_threshold_pct": max(5.0, 200.0 * cv),
                "min_run_median_us": min(medians),
                "max_run_median_us": max(medians),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    summaries = summarize(read_trials(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
