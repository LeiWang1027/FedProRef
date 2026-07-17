#!/usr/bin/env python3
"""Aggregate the minimal weak-class mechanism experiment at seed level."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import torch


GROUPS = ("missing", "weak", "frequent")
METHODS = ("proto_aug", "fedproref")
METHOD_LABELS = {"proto_aug": "ProtoAug", "fedproref": "FedProRef"}
REFINERS = {"proto_aug": "none", "fedproref": "mlp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed-level weak-class mechanism aggregation")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("analysis_outputs/weak_class_mechanism_minimal"))
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--alpha", default="0.01")
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--round-start", type=int, default=96)
    parser.add_argument("--round-end", type=int, default=100)
    parser.add_argument("--total-clients", type=int, default=10)
    parser.add_argument("--selected-clients", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--exp-suffix", default="_weakmech_v2")
    return parser.parse_args()


def load_artifact(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def experiment_name(args, method, training_seed):
    refiner = REFINERS[method]
    return (
        f"plan_{method}_{args.dataset}_a{args.alpha}_"
        f"tc{args.total_clients}_sc{args.selected_clients}_"
        f"le{args.local_epochs}_ps{args.partition_seed}_"
        f"s{training_seed}_ref{refiner}{args.exp_suffix}"
    )


def average(values, context):
    valid = [float(value) for value in values if value is not None]
    if not valid:
        raise ValueError(f"No valid values for {context}")
    return mean(valid), len(valid)


def sample_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.round_start < 1 or args.round_end < args.round_start:
        raise ValueError("Invalid round window")

    expected_rounds = set(range(args.round_start, args.round_end + 1))
    seed_rows = []
    source_artifacts = []

    for method in METHODS:
        for training_seed in args.training_seeds:
            exp_name = experiment_name(args, method, training_seed)
            artifact_path = (
                args.checkpoint_dir
                / f"{exp_name}_{method}_mechanism_metrics.pt"
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Missing mechanism artifact: {artifact_path}")
            payload = load_artifact(artifact_path)
            if payload.get("format") != "fedproref_mechanism_metrics_v2":
                raise ValueError(
                    f"Expected v2 mechanism artifact: {artifact_path}")
            if payload.get("method") != method:
                raise ValueError(f"Method mismatch in {artifact_path}")
            if int(payload.get("partition_seed")) != args.partition_seed:
                raise ValueError(f"Partition seed mismatch in {artifact_path}")
            if int(payload.get("training_seed")) != training_seed:
                raise ValueError(f"Training seed mismatch in {artifact_path}")
            if not payload.get("per_class_saved"):
                raise ValueError(f"Per-class recalls were not saved: {artifact_path}")

            records = [
                record for record in payload.get("mechanism_round_records", [])
                if args.round_start <= int(record["round"]) <= args.round_end
            ]
            observed_rounds = {int(record["round"]) for record in records}
            if observed_rounds != expected_rounds:
                raise ValueError(
                    f"Incomplete round window in {artifact_path}: "
                    f"expected={sorted(expected_rounds)}, observed={sorted(observed_rounds)}")
            per_round_counts = {
                rnd: sum(int(record["round"]) == rnd for record in records)
                for rnd in expected_rounds
            }
            if any(count != args.selected_clients for count in per_round_counts.values()):
                raise ValueError(
                    f"Expected {args.selected_clients} client records per round in "
                    f"{artifact_path}, got {per_round_counts}")

            row = {
                "dataset": args.dataset,
                "alpha": args.alpha,
                "method": METHOD_LABELS[method],
                "method_arg": method,
                "partition_seed": args.partition_seed,
                "training_seed": training_seed,
                "round_start": args.round_start,
                "round_end": args.round_end,
                "client_round_records": len(records),
                "artifact": str(artifact_path),
            }
            for group in GROUPS:
                before, n_before = average(
                    [record[f"{group}_recall_before"] for record in records],
                    f"{method}/seed{training_seed}/{group}/before")
                after, n_after = average(
                    [record[f"{group}_recall_after"] for record in records],
                    f"{method}/seed{training_seed}/{group}/after")
                change, n_change = average(
                    [record[f"{group}_recall_change"] for record in records],
                    f"{method}/seed{training_seed}/{group}/change")
                if not (n_before == n_after == n_change):
                    raise ValueError(
                        f"Inconsistent valid record counts for {method}/{group}")
                row[f"{group}_valid_client_rounds"] = n_change
                row[f"{group}_recall_before_pp"] = before * 100.0
                row[f"{group}_recall_after_pp"] = after * 100.0
                row[f"{group}_recall_change_pp"] = change * 100.0
            seed_rows.append(row)
            source_artifacts.append(str(artifact_path))

    summary_rows = []
    summary = {}
    for method in METHODS:
        method_label = METHOD_LABELS[method]
        method_rows = [row for row in seed_rows if row["method_arg"] == method]
        summary[method_label] = {}
        for group in GROUPS:
            values = [row[f"{group}_recall_change_pp"] for row in method_rows]
            item = {
                "dataset": args.dataset,
                "alpha": args.alpha,
                "method": method_label,
                "group": group,
                "seed_count": len(values),
                "mean_change_pp": mean(values),
                "sample_sd_pp": sample_std(values),
                "min_change_pp": min(values),
                "max_change_pp": max(values),
            }
            summary_rows.append(item)
            summary[method_label][group] = item

    paired_rows = []
    paired_summary = {}
    for group in GROUPS:
        differences = []
        for training_seed in args.training_seeds:
            proto = next(
                row for row in seed_rows
                if row["method_arg"] == "proto_aug"
                and row["training_seed"] == training_seed)
            fed = next(
                row for row in seed_rows
                if row["method_arg"] == "fedproref"
                and row["training_seed"] == training_seed)
            difference = (
                fed[f"{group}_recall_change_pp"]
                - proto[f"{group}_recall_change_pp"]
            )
            differences.append(difference)
            paired_rows.append({
                "dataset": args.dataset,
                "alpha": args.alpha,
                "group": group,
                "partition_seed": args.partition_seed,
                "training_seed": training_seed,
                "fedproref_minus_protoaug_pp": difference,
            })
        paired_summary[group] = {
            "mean_pp": mean(differences),
            "sample_sd_pp": sample_std(differences),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_fields = list(seed_rows[0].keys())
    write_csv(args.output_dir / "seed_level_metrics.csv", seed_rows, seed_fields)
    write_csv(
        args.output_dir / "method_group_summary.csv", summary_rows,
        list(summary_rows[0].keys()))
    write_csv(
        args.output_dir / "paired_method_gain.csv", paired_rows,
        list(paired_rows[0].keys()))

    result_payload = {
        "configuration": {
            "dataset": args.dataset,
            "alpha": args.alpha,
            "partition_seed": args.partition_seed,
            "training_seeds": args.training_seeds,
            "rounds": [args.round_start, args.round_end],
            "aggregation": (
                "mean over participating client-round group macro recalls within "
                "each seed; mean and sample SD over three seed-level values"),
        },
        "summary": summary,
        "paired_fedproref_minus_protoaug": paired_summary,
        "source_artifacts": source_artifacts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8")

    table_lines = [
        "# Weak-class mechanism analysis",
        "",
        (f"CIFAR-100, alpha={args.alpha}, partition seed={args.partition_seed}, "
         f"training seeds={args.training_seeds}, rounds {args.round_start}-{args.round_end}."),
        "Values are recall changes in percentage points, reported as seed-level mean ± sample SD.",
        "",
        "| Method | Missing | Weak | Frequent |",
        "|---|---:|---:|---:|",
    ]
    for method_label in ("ProtoAug", "FedProRef"):
        cells = []
        for group in GROUPS:
            item = summary[method_label][group]
            cells.append(
                f"{item['mean_change_pp']:.2f} ± {item['sample_sd_pp']:.2f}")
        table_lines.append(f"| {method_label} | " + " | ".join(cells) + " |")
    table_lines.extend([
        "",
        "## Paired FedProRef - ProtoAug gain",
        "",
        "| Group | Mean gain (pp) | Sample SD |",
        "|---|---:|---:|",
    ])
    for group in GROUPS:
        item = paired_summary[group]
        table_lines.append(
            f"| {group} | {item['mean_pp']:.2f} | {item['sample_sd_pp']:.2f} |")
    table_lines.append("")
    (args.output_dir / "weak_class_mechanism_table.md").write_text(
        "\n".join(table_lines), encoding="utf-8")

    print(f"Loaded {len(source_artifacts)} completed mechanism artifacts.")
    print(f"Wrote analysis to {args.output_dir}")
    for line in table_lines:
        print(line)


if __name__ == "__main__":
    main()
