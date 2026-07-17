#!/usr/bin/env bash
set -euo pipefail

# Minimal weak-class mechanism analysis plan (6 unique runs):
#   dataset:          CIFAR-100
#   alpha:            0.01
#   methods:          ProtoAug, FedProRef (MLP refiner)
#   partition seed:   42
#   training seeds:   42, 43, 44
#   evaluated rounds: 96-100
#
# The training algorithm is unchanged. Read-only per-class recall metrics are
# collected before and after each participating client's local update.
#
# Usage:
#   bash train_plan_weak_class_mechanism_minimal.sh
#   DRY_RUN=1 bash train_plan_weak_class_mechanism_minimal.sh
#   MAX_JOBS=1 bash train_plan_weak_class_mechanism_minimal.sh
#   GPU_IDS="0 1" bash train_plan_weak_class_mechanism_minimal.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHON_BIN="${PYTHON_BIN:-/home/cherry/miniconda3/envs/fedfm/bin/python}"
export DATASETS="cifar100"
export ALPHAS="0.01"
export METHODS="proto_aug fedproref"
export FEDPROREF_REFINER_TYPES="mlp"
export PARTITION_SEEDS="42"
export TRAINING_SEEDS="42 43 44"
export TOTAL_CLIENTS="10"
export SELECT_CLIENTS="10"
export LOCAL_EPOCHS="10"
export COMM_ROUNDS="100"
export MIN_SAMPLES_PER_CLASS="10"
export WEAK_CLASS_PERCENTILE="10.0"
export MECHANISM_EVAL_START_ROUND="96"
export MECHANISM_EVAL_END_ROUND="100"
export MECHANISM_SAVE_PER_CLASS="1"
export EXP_SUFFIX="_weakmech_v2"
export REQUIRED_ARTIFACT_SUFFIX="_mechanism_metrics.pt"
export SKIP_COMPLETED="1"
export MAX_JOBS="${MAX_JOBS:-2}"
export SAVE_DIR="${SAVE_DIR:-./checkpoints}"

bash "$SCRIPT_DIR/train_plan_cifar_tiny.sh" "$@"

if [[ "${DRY_RUN:-0}" == "1" || "${AUTO_ANALYZE:-1}" != "1" ]]; then
  exit 0
fi

missing_artifacts=0
for training_seed in 42 43 44; do
  for method in proto_aug fedproref; do
    if [[ "$method" == "proto_aug" ]]; then
      refiner_type="none"
    else
      refiner_type="mlp"
    fi
    exp_name="plan_${method}_cifar100_a0.01_tc10_sc10_le10_ps42_s${training_seed}_ref${refiner_type}${EXP_SUFFIX}"
    artifact="${SAVE_DIR}/${exp_name}_${method}_mechanism_metrics.pt"
    if [[ ! -f "$artifact" ]]; then
      printf '[ANALYSIS PENDING] Missing artifact: %s\n' "$artifact" >&2
      missing_artifacts=$((missing_artifacts + 1))
    fi
  done
done

if (( missing_artifacts > 0 )); then
  printf 'Analysis deferred: %s of 6 mechanism artifacts are missing.\n' "$missing_artifacts" >&2
  exit 1
fi

ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-./analysis_outputs/weak_class_mechanism_minimal}"
"$PYTHON_BIN" "$SCRIPT_DIR/analysis_scripts/weak_class_mechanism_analysis.py" \
  --checkpoint-dir "$SAVE_DIR" \
  --output-dir "$ANALYSIS_OUTPUT_DIR"
