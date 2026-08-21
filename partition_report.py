#!/usr/bin/env python3
"""Create evidence for a FedProRef missing-class partition cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


def audit_index_partition(
    train_labels: np.ndarray,
    client_indices,
    num_classes: int,
) -> dict:
    """Validate an index partition and return its client-class count matrix."""
    labels = np.asarray(train_labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("training labels must be one-dimensional")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("training label is outside the declared class range")

    normalized = []
    for client_id, raw_indices in enumerate(client_indices):
        indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= labels.size):
            raise ValueError(f"out-of-range index in client {client_id}")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"within-client duplicate in client {client_id}")
        normalized.append(indices)

    assigned = (
        np.concatenate(normalized) if normalized else np.empty(0, dtype=np.int64)
    )
    if np.unique(assigned).size != assigned.size:
        raise ValueError("cross-client duplicate index")
    expected = np.arange(labels.size, dtype=np.int64)
    if assigned.size != expected.size or not np.array_equal(np.sort(assigned), expected):
        raise ValueError("training samples are not assigned exactly once")

    matrix = np.stack(
        [np.bincount(labels[indices], minlength=num_classes) for indices in normalized]
    ) if normalized else np.zeros((0, num_classes), dtype=np.int64)
    return {
        "matrix": matrix,
        "integrity": {
            "all_training_samples_assigned_once": True,
            "no_out_of_range_indices": True,
            "no_within_client_duplicates": True,
            "no_cross_client_duplicates": True,
        },
    }


def _sample_sd(values: list[int]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _partition_sha256(client_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for client_id, raw_indices in enumerate(client_indices):
        indices = np.asarray(raw_indices, dtype="<i8")
        digest.update(client_id.to_bytes(4, "little", signed=False))
        digest.update(len(indices).to_bytes(8, "little", signed=False))
        digest.update(indices.tobytes(order="C"))
    return digest.hexdigest()


def build_report(
    cache_path: Path,
    dataset: str,
    partition_seed: int,
    num_classes: int,
    upload_threshold: int,
) -> dict:
    with np.load(cache_path, allow_pickle=True) as cache:
        train_labels = np.asarray(cache["train_labels"], dtype=np.int64)
        client_indices = cache["client_indices"]

    audit = audit_index_partition(train_labels, client_indices, num_classes)
    client_counts = audit["matrix"]
    missing_per_client = np.sum(client_counts == 0, axis=1).astype(int).tolist()
    observed_weak_per_client = np.sum(
        (client_counts > 0) & (client_counts < upload_threshold), axis=1
    ).astype(int).tolist()
    client_totals = client_counts.sum(axis=1).astype(int).tolist()
    server_has_anchor = np.any(client_counts >= upload_threshold, axis=0)
    zero_pairs = int(np.sum(client_counts == 0))
    total_pairs = int(client_counts.size)

    return {
        "dataset": dataset,
        "partition_seed": partition_seed,
        "cache_path": str(cache_path.resolve()),
        "num_clients": int(client_counts.shape[0]),
        "num_classes": num_classes,
        "upload_threshold": upload_threshold,
        "total_client_class_pairs": total_pairs,
        "zero_pairs": zero_pairs,
        "zero_pair_percentage": 100.0 * zero_pairs / total_pairs,
        "observed_weak_pairs": int(np.sum(
            (client_counts > 0) & (client_counts < upload_threshold)
        )),
        "strong_pairs": int(np.sum(client_counts >= upload_threshold)),
        "missing_classes_per_client": missing_per_client,
        "missing_classes_per_client_mean": statistics.fmean(missing_per_client),
        "missing_classes_per_client_sample_sd": _sample_sd(missing_per_client),
        "missing_classes_per_client_range": [
            min(missing_per_client), max(missing_per_client)
        ],
        "observed_weak_classes_per_client": observed_weak_per_client,
        "observed_weak_classes_per_client_mean": statistics.fmean(
            observed_weak_per_client
        ),
        "observed_weak_classes_per_client_sample_sd": _sample_sd(
            observed_weak_per_client
        ),
        "observed_weak_classes_per_client_range": [
            min(observed_weak_per_client), max(observed_weak_per_client)
        ],
        "client_total_samples": client_totals,
        "client_total_samples_range": [min(client_totals), max(client_totals)],
        "empty_clients": int(np.sum(client_counts.sum(axis=1) == 0)),
        "global_anchor_classes": int(server_has_anchor.sum()),
        "classes_without_global_anchor": np.flatnonzero(
            ~server_has_anchor
        ).astype(int).tolist(),
        "augmentable_missing_pairs": int(np.sum(
            (client_counts == 0) & server_has_anchor[None, :]
        )),
        "partition_index_sha256": _partition_sha256(client_indices),
        "integrity": audit["integrity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--partition-seed", required=True, type=int)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--upload-threshold", default=10, type=int)
    args = parser.parse_args()

    report = build_report(
        args.cache,
        args.dataset,
        args.partition_seed,
        args.num_classes,
        args.upload_threshold,
    )
    if report["empty_clients"] != 0:
        raise SystemExit("partition evidence failed: empty client found")
    if report["zero_pairs"] == 0:
        raise SystemExit("partition evidence failed: no missing client-class pair")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
