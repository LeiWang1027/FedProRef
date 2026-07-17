#!/usr/bin/env python3
"""Create DirectAnchorAug partition-index caches from local frozen features.

This utility never loads raw images or extracts CLIP features. It applies the
existing data_utils.dirichlet_partition helper to labels in a local global
feature cache and writes the training entry point's atomic cache format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running ``python scripts/...`` puts scripts/ on sys.path, not the repo root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from data_utils import (
    _acquire_cache_lock,
    _load_partition_cache,
    _release_cache_lock,
    _save_partition_cache,
    dirichlet_partition,
)

NUM_CLASSES = {"cifar100": 100, "tinyimagenet": 200}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one deterministic DirectAnchorAug partition cache without downloading data."
    )
    parser.add_argument("--dataset", choices=sorted(NUM_CLASSES), required=True)
    parser.add_argument("--alpha", required=True, help="Dirichlet alpha, preserved verbatim in the cache path")
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--min-require-size", type=int, default=1)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.data_dir) / f"{args.dataset}_clip_cache"
    global_caches = sorted(cache_dir.glob("*_global_features.npz"))
    if len(global_caches) != 1:
        raise SystemExit(
            f"Expected exactly one local global feature cache in {cache_dir}; "
            f"found {len(global_caches)}. This utility will not download or extract features."
        )

    target = (
        cache_dir / "partitions" /
        f"{args.dataset}_alpha{args.alpha}_c{args.num_clients}_"
        f"min{args.min_require_size}_ps{args.partition_seed}.npz"
    )
    if target.exists():
        _load_partition_cache(str(target), args.num_clients)
        print(f"[PARTITION READY] {target}")
        return 0
    if args.dry_run:
        print(f"[PLAN PARTITION] {global_caches[0]} -> {target}")
        return 0

    # Read labels only. allow_pickle=False prevents object deserialization.
    with np.load(global_caches[0], allow_pickle=False) as data:
        labels = np.asarray(data["train_labels"], dtype=np.int64)
    num_classes = NUM_CLASSES[args.dataset]
    if labels.size == 0 or labels.min() < 0 or labels.max() >= num_classes:
        raise SystemExit(f"Unexpected labels in {global_caches[0]} for {args.dataset}")

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_file, lock_fd = _acquire_cache_lock(str(target))
    try:
        if target.exists():
            _load_partition_cache(str(target), args.num_clients)
            print(f"[PARTITION READY] {target}")
            return 0
        indices = dirichlet_partition(
            labels=labels, num_clients=args.num_clients, alpha=float(args.alpha),
            num_classes=num_classes, min_require_size=args.min_require_size,
            seed=args.partition_seed,
        )
        _save_partition_cache(str(target), indices)
    finally:
        _release_cache_lock(lock_file, lock_fd)

    _load_partition_cache(str(target), args.num_clients)
    counts = [len(item) for item in indices]
    print(
        f"[PARTITION CREATED] {target} clients={args.num_clients} "
        f"samples={sum(counts)} min_client_samples={min(counts)} "
        f"max_client_samples={max(counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
