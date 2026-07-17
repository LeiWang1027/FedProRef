#!/usr/bin/env bash
set -euo pipefail

# Training plan: FedProRef refiner=rf experiment
# datasets: cifar10, cifar100, tinyimagenet
# alpha: 0.01 0.03 0.05 0.07 0.09 0.1 0.3 0.5
# total clients: 10, selected clients: 10, local epochs: 10, seeds: 42 43 44
# method: fedproref
# refiner: rf
#
# Usage:
#   bash train_plan_fedproref_refrf.sh
#   DRY_RUN=1 bash train_plan_fedproref_refrf.sh
#   MAX_JOBS=3 bash train_plan_fedproref_refrf.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/cherry/miniconda3/envs/fedfm/bin/python}"
DATASETS=(cifar10 cifar100 tinyimagenet)
ALPHAS=(0.01 0.03 0.05 0.07 0.09 0.1 0.3 0.5)

TOTAL_CLIENTS=10
SELECT_CLIENTS=10
LOCAL_EPOCHS=10
SEEDS=(42 43 44)
PARTITION_SEED="${PARTITION_SEED:-42}"
MAX_JOBS="${MAX_JOBS:-2}"

METHODS_ARRAY=(fedproref)
FEDPROREF_REFINER_TYPES=(rf)
DATA_DIR="${DATA_DIR:-./data}"
LOG_DIR="${LOG_DIR:-./logs}"
SAVE_DIR="${SAVE_DIR:-./checkpoints}"

BACKBONE="${BACKBONE:-ViT-B-16}"
PRETRAINED="${PRETRAINED:-./pretrain_path/old_open_clip_model.safetensors}"
FEAT_DIM="${FEAT_DIM:-512}"
HEAD_TYPE="${HEAD_TYPE:-linear}"
HEAD_HIDDEN="${HEAD_HIDDEN:-512}"

COMM_ROUNDS="${COMM_ROUNDS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
BETA_HEAD="${BETA_HEAD:-0.2}"

NUM_MODES="${NUM_MODES:-1}"
PROTO_MERGE_THRESHOLD="${PROTO_MERGE_THRESHOLD:-0.90}"
PROTO_MERGE_THRESHOLD_MIN="${PROTO_MERGE_THRESHOLD_MIN:-0.70}"
PROTO_MERGE_THRESHOLD_MAX="${PROTO_MERGE_THRESHOLD_MAX:-0.98}"
PROTO_MERGE_THRESHOLD_STEP="${PROTO_MERGE_THRESHOLD_STEP:-0.03}"
PROTO_MERGE_TARGET_LOW="${PROTO_MERGE_TARGET_LOW:-0.04}"
PROTO_MERGE_TARGET_HIGH="${PROTO_MERGE_TARGET_HIGH:-0.10}"
PROTO_MERGE_THRESHOLD_LR="${PROTO_MERGE_THRESHOLD_LR:-0.05}"
PROTO_MERGE_TEMPERATURE="${PROTO_MERGE_TEMPERATURE:-0.05}"
PROTO_MERGE_LEARN_STEPS="${PROTO_MERGE_LEARN_STEPS:-10}"
PROTO_MERGE_TAU_REG="${PROTO_MERGE_TAU_REG:-0.1}"
PROTO_MERGE_ACC_WEIGHT="${PROTO_MERGE_ACC_WEIGHT:-0.5}"
PROTO_MERGE_ACC_DROP_TOLERANCE="${PROTO_MERGE_ACC_DROP_TOLERANCE:-0.3}"
PROTO_MERGE_ACC_GAIN_TOLERANCE="${PROTO_MERGE_ACC_GAIN_TOLERANCE:-0.2}"

PROPOSAL_SIGMA="${PROPOSAL_SIGMA:-0.05}"
REFINER_HIDDEN="${REFINER_HIDDEN:-512}"
REFINER_LAYERS="${REFINER_LAYERS:-3}"
RF_STEPS="${RF_STEPS:-4}"
REFINER_PRETRAIN_EPOCHS="${REFINER_PRETRAIN_EPOCHS:-300}"
REFINER_FINETUNE_EPOCHS="${REFINER_FINETUNE_EPOCHS:-50}"
REFINER_LR="${REFINER_LR:-1e-3}"
CAL_EVERY="${CAL_EVERY:-10}"
W_REG="${W_REG:-0.01}"
W_PROTO="${W_PROTO:-0.1}"

GEN_PER_CLASS="${GEN_PER_CLASS:-500}"
AUG_GEN_PER_CLASS="${AUG_GEN_PER_CLASS:-100}"
WEAK_CLASS_PERCENTILE="${WEAK_CLASS_PERCENTILE:-10.0}"
CAL_BUDGET="${CAL_BUDGET:-500}"
CAL_EPOCHS="${CAL_EPOCHS:-20}"
CAL_LR="${CAL_LR:-1e-3}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-1}"
MIN_SAMPLES_PER_CLASS="${MIN_SAMPLES_PER_CLASS:-10}"

run_one() {
  local dataset="$1"
  local alpha="$2"
  local seed="$3"
  local method="$4"
  local refiner_type="$5"
  local exp_name="plan_${method}_${dataset}_a${alpha}_tc${TOTAL_CLIENTS}_sc${SELECT_CLIENTS}_le${LOCAL_EPOCHS}_ps${PARTITION_SEED}_s${seed}_ref${refiner_type}"

  local cmd=(
    "$PYTHON_BIN" "federated_loop.py"
    --dataset "$dataset"
    --data_dir "$DATA_DIR"
    --num_clients "$TOTAL_CLIENTS"
    --select_clients "$SELECT_CLIENTS"
    --alpha "$alpha"
    --min_require_size "$MIN_REQUIRE_SIZE"
    --min_samples_per_class "$MIN_SAMPLES_PER_CLASS"
    --backbone "$BACKBONE"
    --pretrained "$PRETRAINED"
    --feat_dim "$FEAT_DIM"
    --head_type "$HEAD_TYPE"
    --head_hidden "$HEAD_HIDDEN"
    --comm_rounds "$COMM_ROUNDS"
    --local_epochs "$LOCAL_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --beta_head "$BETA_HEAD"
    --num_modes "$NUM_MODES"
    --proto_merge_threshold "$PROTO_MERGE_THRESHOLD"
    --proto_merge_threshold_min "$PROTO_MERGE_THRESHOLD_MIN"
    --proto_merge_threshold_max "$PROTO_MERGE_THRESHOLD_MAX"
    --proto_merge_threshold_step "$PROTO_MERGE_THRESHOLD_STEP"
    --proto_merge_target_low "$PROTO_MERGE_TARGET_LOW"
    --proto_merge_target_high "$PROTO_MERGE_TARGET_HIGH"
    --proto_merge_threshold_lr "$PROTO_MERGE_THRESHOLD_LR"
    --proto_merge_temperature "$PROTO_MERGE_TEMPERATURE"
    --proto_merge_learn_steps "$PROTO_MERGE_LEARN_STEPS"
    --proto_merge_tau_reg "$PROTO_MERGE_TAU_REG"
    --proto_merge_acc_weight "$PROTO_MERGE_ACC_WEIGHT"
    --proto_merge_acc_drop_tolerance "$PROTO_MERGE_ACC_DROP_TOLERANCE"
    --proto_merge_acc_gain_tolerance "$PROTO_MERGE_ACC_GAIN_TOLERANCE"
    --proposal_sigma "$PROPOSAL_SIGMA"
    --refiner_type "$refiner_type"
    --refiner_hidden "$REFINER_HIDDEN"
    --refiner_layers "$REFINER_LAYERS"
    --rf_steps "$RF_STEPS"
    --refiner_pretrain_epochs "$REFINER_PRETRAIN_EPOCHS"
    --refiner_finetune_epochs "$REFINER_FINETUNE_EPOCHS"
    --refiner_lr "$REFINER_LR"
    --cal_every "$CAL_EVERY"
    --w_reg "$W_REG"
    --w_proto "$W_PROTO"
    --gen_per_class "$GEN_PER_CLASS"
    --aug_gen_per_class "$AUG_GEN_PER_CLASS"
    --weak_class_percentile "$WEAK_CLASS_PERCENTILE"
    --cal_budget "$CAL_BUDGET"
    --cal_epochs "$CAL_EPOCHS"
    --cal_lr "$CAL_LR"
    --method "$method"
    --seed "$seed"
    --partition_seed "$PARTITION_SEED"
    --device auto
    --log_dir "$LOG_DIR"
    --save_dir "$SAVE_DIR"
    --exp_name "$exp_name"
  )

  printf '\n============================================================\n'
  printf 'Dataset=%s alpha=%s method=%s total-client=%s select-client=%s local-epoch=%s seed=%s refiner=%s\n' \
    "$dataset" "$alpha" "$method" "$TOTAL_CLIENTS" "$SELECT_CLIENTS" "$LOCAL_EPOCHS" "$seed" "$refiner_type"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n============================================================\n'

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}"
}

running_jobs=0
failures=0

wait_for_slot() {
  while (( running_jobs >= MAX_JOBS )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    running_jobs=$((running_jobs - 1))
  done
}

for dataset in "${DATASETS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for method in "${METHODS_ARRAY[@]}"; do
        if [[ "$method" == "fedproref" ]]; then
          refiner_types=("${FEDPROREF_REFINER_TYPES[@]}")
        else
          refiner_types=(none)
        fi

        for refiner_type in "${refiner_types[@]}"; do
          if [[ "${DRY_RUN:-0}" == "1" ]]; then
            run_one "$dataset" "$alpha" "$seed" "$method" "$refiner_type"
          else
            wait_for_slot
            run_one "$dataset" "$alpha" "$seed" "$method" "$refiner_type" &
            running_jobs=$((running_jobs + 1))
          fi
        done
      done
    done
  done
done

while (( running_jobs > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  running_jobs=$((running_jobs - 1))
done

if (( failures > 0 )); then
  echo "${failures} experiment(s) failed." >&2
  exit 1
fi
