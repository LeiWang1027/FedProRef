#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar100}"
ALPHA="${2:-0.3}"
METHOD="${3:-fedproref}"
if [[ $# -ge 3 ]]; then
  shift 3
else
  set --
fi

cd "$(dirname "$0")"

python run.py \
  --dataset "$DATASET" \
  --alpha "$ALPHA" \
  --method "$METHOD" \
  --seed 42 \
  --repeats 3 \
  "$@"
