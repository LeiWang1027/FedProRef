#!/usr/bin/env python3
"""Parse and summarize the fixed RN50 backbone-robustness experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


RUN_HEADERS = [
    "dataset", "backbone", "pretrained", "feature_dim", "min_require_size",
    "partition_seed", "training_seed", "method", "best_accuracy", "best_round",
    "checkpoint_hash", "cache_file", "log_file", "run_status",
]
AGGREGATE_HEADERS = [
    "dataset", "backbone", "min_require_size", "method", "n",
    "training_seeds", "complete", "mean_accuracy", "sample_sd",
]
DELTA_HEADERS = [
    "dataset", "backbone", "min_require_size", "partition_seed",
    "training_seed", "control_method", "fedproref_accuracy",
    "control_accuracy", "delta", "win", "mean_delta", "sample_sd", "win_count",
]
EXPECTED_SEEDS = {42, 43, 44}
CONTROL_METHODS = ("proto_aug", "direct_anchor_aug", "fedavg")


def _cell(row: dict) -> tuple:
    return (
        str(row["dataset"]), int(row["min_require_size"]),
        int(row["partition_seed"]), int(row["training_seed"]), str(row["method"]),
    )


def _validate_unique_cells(rows: Iterable[dict]) -> list[dict]:
    rows = list(rows)
    seen = set()
    for row in rows:
        cell = _cell(row)
        if cell in seen:
            raise ValueError(f"duplicate dataset/floor/partition-seed/training-seed/method cell: {cell}")
        seen.add(cell)
    return rows


def _parse_best(log_text: str) -> tuple[float, int]:
    round_matches = re.findall(
        r"^Round\s+(\d+)\s*\|[^\n]*?\bacc:\s*([0-9]+(?:\.[0-9]+)?)",
        log_text,
        flags=re.MULTILINE,
    )
    if round_matches:
        best_round, best_accuracy = max(
            round_matches, key=lambda item: float(item[1]))
        return float(best_accuracy), int(best_round)

    best_matches = re.findall(
        r"Best:\s*Round\s+(\d+)\s*\|\s*Acc=([0-9]+(?:\.[0-9]+)?)%",
        log_text,
    )
    if best_matches:
        round_text, accuracy_text = best_matches[-1]
        return float(accuracy_text), int(round_text)
    raise ValueError("console log contains no accuracy records")


def _extract(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def parse_completed_runs(results_dir: str | Path) -> list[dict]:
    """Read strictly completed run directories and reject duplicate cells."""
    results_dir = Path(results_dir)
    rows = []
    if not results_dir.exists():
        return rows

    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        marker = run_dir / ".done"
        config_file = run_dir / "config.json"
        console_file = run_dir / "console.log"
        if not (marker.is_file() and config_file.is_file() and console_file.is_file()):
            continue

        with config_file.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        required = {
            "dataset", "backbone", "pretrained", "min_require_size",
            "partition_seed", "training_seed", "method",
        }
        missing = required.difference(config)
        if missing:
            raise ValueError(f"{run_dir}: config.json is missing {sorted(missing)}")

        log_text = console_file.read_text(encoding="utf-8", errors="replace")
        best_accuracy, best_round = _parse_best(log_text)
        feature_dim_text = _extract(r"^\s*Feature dimension:\s*(\d+)\s*$", log_text)
        if not feature_dim_text:
            feature_dim_text = _extract(r"^\s*feat_dim\s*=\s*(\d+)\s*$", log_text)
        if not feature_dim_text:
            raise ValueError(f"{run_dir}: feature dimension is absent from console.log")

        rows.append({
            "dataset": str(config["dataset"]),
            "backbone": str(config["backbone"]),
            "pretrained": str(config["pretrained"]),
            "feature_dim": int(feature_dim_text),
            "min_require_size": int(config["min_require_size"]),
            "partition_seed": int(config["partition_seed"]),
            "training_seed": int(config["training_seed"]),
            "method": str(config["method"]),
            "best_accuracy": best_accuracy,
            "best_round": best_round,
            "checkpoint_hash": _extract(
                r"^\s*Checkpoint SHA-256:\s*(\S+)\s*$", log_text, "unknown"),
            "cache_file": _extract(
                r"^\s*feature_cache_file\s*=\s*(.+?)\s*$", log_text),
            "log_file": str(console_file.resolve()),
            "run_status": "complete",
        })
    return _validate_unique_cells(rows)


def aggregate_runs(rows: Iterable[dict]) -> list[dict]:
    """Aggregate accuracy by dataset/floor/method using sample SD."""
    rows = _validate_unique_cells(rows)
    groups = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]), str(row.get("backbone", "RN50")),
            int(row["min_require_size"]), str(row["method"]),
        )
        groups[key].append(row)

    output = []
    for (dataset, backbone, floor, method), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["training_seed"]))
        seeds = [int(row["training_seed"]) for row in group]
        accuracies = [float(row["best_accuracy"]) for row in group]
        output.append({
            "dataset": dataset,
            "backbone": backbone,
            "min_require_size": floor,
            "method": method,
            "n": len(group),
            "training_seeds": ";".join(map(str, seeds)),
            "complete": set(seeds) == EXPECTED_SEEDS and len(seeds) == len(EXPECTED_SEEDS),
            "mean_accuracy": statistics.fmean(accuracies),
            "sample_sd": statistics.stdev(accuracies) if len(accuracies) >= 2 else "",
        })
    return output


def paired_deltas(rows: Iterable[dict]) -> list[dict]:
    """Compute FedProRef-minus-control differences only within identical cells."""
    rows = _validate_unique_cells(rows)
    lookup = {_cell(row): row for row in rows}
    grouped = defaultdict(list)

    for row in rows:
        if row["method"] != "fedproref":
            continue
        dataset, floor, partition_seed, seed, _ = _cell(row)
        for control in CONTROL_METHODS:
            control_row = lookup.get((dataset, floor, partition_seed, seed, control))
            if control_row is None:
                continue
            grouped[(dataset, str(row.get("backbone", "RN50")), floor,
                     partition_seed, control)].append({
                "training_seed": seed,
                "fedproref_accuracy": float(row["best_accuracy"]),
                "control_accuracy": float(control_row["best_accuracy"]),
                "delta": float(row["best_accuracy"]) - float(control_row["best_accuracy"]),
            })

    output = []
    for (dataset, backbone, floor, partition_seed, control), pairs in sorted(grouped.items()):
        pairs.sort(key=lambda pair: pair["training_seed"])
        deltas = [pair["delta"] for pair in pairs]
        mean_delta = statistics.fmean(deltas)
        sample_sd = statistics.stdev(deltas) if len(deltas) >= 2 else ""
        wins = sum(delta > 0 for delta in deltas)
        for pair in pairs:
            output.append({
                "dataset": dataset,
                "backbone": backbone,
                "min_require_size": floor,
                "partition_seed": partition_seed,
                "training_seed": pair["training_seed"],
                "control_method": control,
                "fedproref_accuracy": pair["fedproref_accuracy"],
                "control_accuracy": pair["control_accuracy"],
                "delta": pair["delta"],
                "win": pair["delta"] > 0,
                "mean_delta": mean_delta,
                "sample_sd": sample_sd,
                "win_count": wins,
            })
    return output


def _write_csv(path: Path, headers: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    return "" if value == "" else f"{float(value):.4f}"


def _markdown_summary(aggregates: list[dict], deltas: list[dict]) -> str:
    lines = [
        "# OpenCLIP RN50 Backbone Robustness", "",
        "Accuracy values are best global test accuracy within 100 rounds. "
        "SD is the sample standard deviation over training seeds 42, 43, and 44.", "",
    ]
    cells = sorted({(row["dataset"], row["min_require_size"]) for row in aggregates})
    for dataset, floor in cells:
        lines.extend([
            f"## {dataset}: min_require_size={floor}", "",
            "| Method | n | Seeds | Complete | Mean accuracy | Sample SD |",
            "|---|---:|---|:---:|---:|---:|",
        ])
        for row in aggregates:
            if (row["dataset"], row["min_require_size"]) != (dataset, floor):
                continue
            lines.append(
                f"| {row['method']} | {row['n']} | {row['training_seeds']} | "
                f"{'yes' if row['complete'] else 'no'} | {_fmt(row['mean_accuracy'])} | "
                f"{_fmt(row['sample_sd'])} |"
            )
        lines.extend([
            "", "### Paired FedProRef differences", "",
            "| Control | Seed 42 | Seed 43 | Seed 44 | Mean delta | Sample SD | Wins |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for control in CONTROL_METHODS:
            subset = [row for row in deltas
                      if row["dataset"] == dataset
                      and row["min_require_size"] == floor
                      and row["control_method"] == control]
            if not subset:
                continue
            by_seed = {row["training_seed"]: row["delta"] for row in subset}
            representative = subset[0]
            lines.append(
                f"| {control} | {_fmt(by_seed.get(42, ''))} | "
                f"{_fmt(by_seed.get(43, ''))} | {_fmt(by_seed.get(44, ''))} | "
                f"{_fmt(representative['mean_delta'])} | "
                f"{_fmt(representative['sample_sd'])} | "
                f"{representative['win_count']}/{len(subset)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(rows: Iterable[dict], output_dir: str | Path) -> None:
    rows = _validate_unique_cells(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_runs(rows)
    deltas = paired_deltas(rows)
    _write_csv(output_dir / "runs.csv", RUN_HEADERS, rows)
    _write_csv(output_dir / "aggregate.csv", AGGREGATE_HEADERS, aggregates)
    _write_csv(output_dir / "paired_deltas.csv", DELTA_HEADERS, deltas)
    (output_dir / "summary.md").write_text(
        _markdown_summary(aggregates, deltas), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = parse_completed_runs(args.results_dir)
    write_outputs(rows, args.output_dir)
    print(f"Summarized {len(rows)} completed runs into {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
