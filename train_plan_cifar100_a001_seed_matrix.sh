#!/usr/bin/env bash
set -euo pipefail

# Experiment matrix (18 unique runs):
#   dataset:          CIFAR-100
#   alpha:            0.01
#   methods:          ProtoAug, FedProRef (MLP refiner)
#   partition seeds:  42, 43, 44
#   training seeds:   42, 43, 44
#
# Completed runs are skipped by default. The underlying runner also takes an
# exclusive per-experiment lock so concurrent invocations cannot duplicate work.
#
# Usage:
#   bash train_plan_cifar100_a001_seed_matrix.sh
#   DRY_RUN=1 bash train_plan_cifar100_a001_seed_matrix.sh
#   MAX_JOBS=2 bash train_plan_cifar100_a001_seed_matrix.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASETS="cifar100"
export ALPHAS="0.01"
export METHODS="proto_aug fedproref"
export FEDPROREF_REFINER_TYPES="mlp"
export PARTITION_SEEDS="42 43 44"
export TRAINING_SEEDS="42 43 44"
export MAX_JOBS="${MAX_JOBS:-2}"
export SKIP_COMPLETED=1

exec bash "$SCRIPT_DIR/train_plan_cifar_tiny.sh" "$@"
