#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOTAL_CLIENTS=50
export SELECT_CLIENTS=10
export LOCAL_EPOCHS=10
export COMM_ROUNDS=100
export METHODS="fedproref"
export FEDPROREF_REFINER_TYPES="mlp"

exec bash "$SCRIPT_DIR/train_plan_cifar_tiny.sh" "$@"
