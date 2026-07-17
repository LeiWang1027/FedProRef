#!/usr/bin/env python3
"""Summarize completed DirectAnchorAug runs without changing manuscript files."""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/direct_anchor_aug_experiment_manifest.csv"
OUT_CSV = ROOT / "results/direct_anchor_aug_summary.csv"
OUT_MD = ROOT / "results/direct_anchor_aug_summary.md"
OUT_TEX = ROOT / "results/direct_anchor_aug_table_row.tex"


def best_accuracy(metrics_path: Path) -> float:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "acc" not in rows[0]:
        raise ValueError(f"missing acc column: {metrics_path}")
    return max(float(row["acc"]) for row in rows)


def comparator_values(dataset: str, alpha: str) -> dict[str, float]:
    values: dict[str, list[float]] = {"proto_aug": [], "fedproref": []}
    for candidate in (ROOT / "results/selected_runs.csv", ROOT / "results/completed_experiment_runs.csv"):
        if not candidate.exists():
            continue
        with candidate.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method = row.get("method", "").lower()
                if method not in values or row.get("dataset") != dataset or str(row.get("alpha")) != alpha:
                    continue
                for field in ("best_acc", "acc", "best_accuracy"):
                    if row.get(field):
                        values[method].append(float(row[field]))
                        break
    return {method: (sum(items) / len(items) if items else float("nan")) for method, items in values.items()}


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    if not rows or len(rows) % 3:
        raise SystemExit(
            f"manifest must contain complete three-seed settings, found {len(rows)} runs"
        )
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        run_dir = ROOT / "results/direct_anchor_aug" / row["dataset"] / f"alpha_{row['alpha']}" / f"seed_{row['training_seed']}"
        required = [run_dir / name for name in ("config.json", "metrics.csv", "run_metadata.json", "stdout.log")]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit("incomplete run: " + ", ".join(missing))
        grouped.setdefault((row["dataset"], row["alpha"]), []).append((int(row["training_seed"]), best_accuracy(run_dir / "metrics.csv")))
    summary = []
    setting_means = []
    for (dataset, alpha), items in sorted(grouped.items()):
        if sorted(seed for seed, _ in items) != [42, 43, 44]:
            raise SystemExit(f"expected seeds 42/43/44 for {dataset} alpha={alpha}")
        values = [value for _, value in sorted(items)]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        compare = comparator_values(dataset, alpha)
        summary.append({"dataset": dataset, "alpha": alpha, "seed_42": values[0], "seed_43": values[1], "seed_44": values[2], "mean": mean, "sample_sd_ddof1": sd, "protoaug_mean": compare["proto_aug"], "fedproref_mean": compare["fedproref"], "delta_vs_protoaug": mean - compare["proto_aug"] if not math.isnan(compare["proto_aug"]) else float("nan"), "delta_vs_fedproref": mean - compare["fedproref"] if not math.isnan(compare["fedproref"]) else float("nan")})
        setting_means.append(mean)
    overall = sum(setting_means) / len(setting_means)
    fields = list(summary[0])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# DirectAnchorAug summary", "", "Single-seed values are retained; SD uses ddof=1.", "", "| Setting | Seed 42 | Seed 43 | Seed 44 | Mean | SD | Δ ProtoAug | Δ FedProRef |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} α={row['alpha']} | {fmt(row['seed_42'])} | {fmt(row['seed_43'])} | {fmt(row['seed_44'])} | {fmt(row['mean'])} | {fmt(row['sample_sd_ddof1'])} | {fmt(row['delta_vs_protoaug'])} | {fmt(row['delta_vs_fedproref'])} |")
    lines.extend(["", f"Setting-average of means: {overall:.4f}", "", "Comparator deltas are descriptive only; blank values mean no matching current-result rows were found."])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex_values = " / ".join(f"{row['mean']:.2f} $\\pm$ {row['sample_sd_ddof1']:.2f}" for row in summary)
    OUT_TEX.write_text(f"DirectAnchorAug & {tex_values} & {overall:.2f} " + r"\\" + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
