#!/usr/bin/env python3
"""Summarize best global accuracy for the focused missing-class experiment."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(
    r"^mcstress_(cifar100|tinyimagenet)_a001_ps(\d+)_ts(\d+)_"
    r"(proto_aug|fedproref)$"
)
BEST_RE = re.compile(
    r"Best:\s*Round\s+(\d+)\s*\|\s*Acc=([0-9.]+)%"
)


def collect_runs(results_root: Path) -> list[dict]:
    rows = []
    if not results_root.exists():
        return rows
    for run_dir in sorted(results_root.iterdir()):
        match = RUN_RE.match(run_dir.name)
        log_path = run_dir / "console.log"
        if not match or not (run_dir / ".done").is_file() or not log_path.is_file():
            continue
        matches = BEST_RE.findall(log_path.read_text(encoding="utf-8", errors="replace"))
        if not matches:
            continue
        best_round, best_acc = matches[-1]
        rows.append({
            "run_id": run_dir.name,
            "dataset": match.group(1),
            "partition_seed": int(match.group(2)),
            "training_seed": int(match.group(3)),
            "method": match.group(4),
            "best_round": int(best_round),
            "best_acc": float(best_acc),
            "log_path": str(log_path.resolve()),
        })
    return rows


def partition_summaries(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["partition_seed"], row["method"])].append(
            row["best_acc"]
        )
    output = []
    for (dataset, partition_seed, method), values in sorted(groups.items()):
        output.append({
            "dataset": dataset,
            "partition_seed": partition_seed,
            "method": method,
            "n_training_seeds": len(values),
            "mean_best_acc": statistics.fmean(values),
            "sample_sd_best_acc": (
                statistics.stdev(values) if len(values) > 1 else None
            ),
        })
    return output


def across_partition_summaries(partition_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in partition_rows:
        groups[(row["dataset"], row["method"])].append(row["mean_best_acc"])
    output = []
    for (dataset, method), values in sorted(groups.items()):
        output.append({
            "dataset": dataset,
            "method": method,
            "n_partitions": len(values),
            "across_partition_mean_best_acc": statistics.fmean(values),
            "sample_sd_partition_means": (
                statistics.stdev(values) if len(values) > 1 else None
            ),
        })
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_root / "summary"

    runs = collect_runs(args.results_root)
    partitions = partition_summaries(runs)
    across = across_partition_summaries(partitions)
    write_csv(output_dir / "runs.csv", runs, [
        "run_id", "dataset", "partition_seed", "training_seed", "method",
        "best_round", "best_acc", "log_path",
    ])
    write_csv(output_dir / "partition_summary.csv", partitions, [
        "dataset", "partition_seed", "method", "n_training_seeds",
        "mean_best_acc", "sample_sd_best_acc",
    ])
    write_csv(output_dir / "across_partitions.csv", across, [
        "dataset", "method", "n_partitions",
        "across_partition_mean_best_acc", "sample_sd_partition_means",
    ])
    print(f"Completed runs: {len(runs)}/24")
    print(f"Summary directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
