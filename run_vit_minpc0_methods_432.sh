#!/usr/bin/env bash
set -euo pipefail

PLAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FEDPROREF_PROJECT:-$PLAN_ROOT}"
DATA_ROOT="${FEDPROREF_DATA_ROOT:-$PROJECT/data}"
RESULTS_ROOT="${FEDPROREF_VIT_MINPC0_RESULTS_ROOT:-$PROJECT/results/vit_minpc0_methods_432}"
LEGACY_RESULTS_ROOT="${FEDPROREF_LEGACY_RESULTS_ROOT:-$PROJECT/results}"
CONDA_ENV="${FEDPROREF_CONDA_ENV:-fedfm}"
PYTHON_BIN="${FEDPROREF_PYTHON_BIN:-}"
VIT_PRETRAINED="${FEDPROREF_VIT_PRETRAINED:-$PROJECT/pretrain_path/old_open_clip_model.safetensors}"
VIT_PRETRAINED_SHA256="4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f"
MAX_JOBS="${MAX_JOBS:-3}"

export PYTHONDONTWRITEBYTECODE=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash run_vit_minpc0_methods_432.sh plan
  bash run_vit_minpc0_methods_432.sh missing
  bash run_vit_minpc0_methods_432.sh status
  bash run_vit_minpc0_methods_432.sh command RUN_ID
  bash run_vit_minpc0_methods_432.sh all

Matrix:
  CIFAR-10/CIFAR-100: 8 alpha values x 3 methods x 3 training seeds.
  TinyImageNet: alpha in {0.01, 0.03, 0.05} x 3 methods x 3 seeds.
  total_clients=10 for every dataset, for 171 runs in total.

Fixed settings:
  min_require_size=0, partition_seed=42, total_clients=10, select_clients=10,
  comm_rounds=100, local_epochs=10, backbone=ViT-B-16.

The all action runs up to three experiments concurrently and is resumable. It
skips both valid results under RESULTS_ROOT and the 12 audited legacy mcstress
runs that match this protocol. Set MAX_JOBS=1, 2, or 3 to change concurrency.

Environment overrides:
  FEDPROREF_PROJECT, FEDPROREF_DATA_ROOT,
  FEDPROREF_VIT_MINPC0_RESULTS_ROOT, FEDPROREF_LEGACY_RESULTS_ROOT,
  FEDPROREF_CONDA_ENV, FEDPROREF_PYTHON_BIN, FEDPROREF_VIT_PRETRAINED
EOF
}

alpha_tag() {
  case "$1" in
    0.01) printf 'a001' ;;
    0.03) printf 'a003' ;;
    0.05) printf 'a005' ;;
    0.07) printf 'a007' ;;
    0.09) printf 'a009' ;;
    0.10) printf 'a010' ;;
    0.30) printf 'a030' ;;
    0.50) printf 'a050' ;;
    *) echo "Unsupported alpha: $1" >&2; return 2 ;;
  esac
}

matrix() {
  local dataset alpha training_seed method tag run_id
  local total_clients=10
  local -a alphas=()
  for dataset in cifar10 cifar100 tinyimagenet; do
    if [[ "$dataset" == tinyimagenet ]]; then
      alphas=(0.01 0.03 0.05)
    else
      alphas=(0.01 0.03 0.05 0.07 0.09 0.10 0.30 0.50)
    fi
    for alpha in "${alphas[@]}"; do
      tag="$(alpha_tag "$alpha")"
      for training_seed in 42 43 44; do
        for method in fedproref proto_aug direct_anchor_aug; do
          run_id="vit_minpc0_${dataset}_${tag}_c${total_clients}_sc10_ps42_ts${training_seed}_${method}"
          printf '%s\t%s\t%s\t%s\t10\t42\t%s\t%s\n' \
            "$run_id" "$dataset" "$alpha" "$total_clients" \
            "$training_seed" "$method"
        done
      done
    done
  done
}

python_prefix() {
  if [[ -n "$PYTHON_BIN" ]]; then
    printf '%s\0' env -u ALL_PROXY -u all_proxy "$PYTHON_BIN"
  else
    printf '%s\0' env -u ALL_PROXY -u all_proxy \
      conda run --no-capture-output -n "$CONDA_ENV" python
  fi
}

build_command() {
  local run_id="$1" dataset="$2" alpha="$3" total_clients="$4"
  local select_clients="$5" partition_seed="$6" training_seed="$7" method="$8"
  local run_dir="$RESULTS_ROOT/$run_id"
  local refiner_type=none
  [[ "$method" == fedproref ]] && refiner_type=mlp

  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)
  COMMAND=(
    "${prefix[@]}" "$PROJECT/federated_loop.py"
    --dataset "$dataset"
    --data_dir "$DATA_ROOT"
    --num_clients "$total_clients"
    --select_clients "$select_clients"
    --alpha "$alpha"
    --min_require_size 0
    --min_samples_per_class 10
    --backbone ViT-B-16
    --pretrained "$VIT_PRETRAINED"
    --feat_dim 512
    --head_type linear
    --head_hidden 512
    --comm_rounds 100
    --local_epochs 10
    --batch_size 64
    --lr 0.001
    --num_modes 1
    --proto_merge_threshold 0.90
    --proto_merge_threshold_min 0.70
    --proto_merge_threshold_max 0.98
    --proposal_sigma 0.05
    --refiner_type "$refiner_type"
    --refiner_hidden 512
    --refiner_layers 3
    --refiner_pretrain_epochs 300
    --refiner_finetune_epochs 50
    --refiner_lr 0.001
    --cal_every 10
    --w_proto 0.1
    --w_reg 0.01
    --gen_per_class 500
    --aug_gen_per_class 100
    --weak_class_percentile 10.0
    --method "$method"
    --seed "$training_seed"
    --partition_seed "$partition_seed"
    --device auto
    --log_dir "$run_dir/logs"
    --save_dir "$run_dir/checkpoints"
    --exp_name "$run_id"
  )
}

find_run() {
  local wanted="$1" row
  while IFS= read -r row; do
    if [[ "${row%%$'\t'*}" == "$wanted" ]]; then
      printf '%s\n' "$row"
      return 0
    fi
  done < <(matrix)
  echo "Unknown run ID: $wanted" >&2
  return 2
}

print_command() {
  local row run_id dataset alpha total_clients select_clients
  local partition_seed training_seed method
  row="$(find_run "$1")"
  IFS=$'\t' read -r run_id dataset alpha total_clients select_clients \
    partition_seed training_seed method <<<"$row"
  build_command "$run_id" "$dataset" "$alpha" "$total_clients" \
    "$select_clients" "$partition_seed" "$training_seed" "$method"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
}

current_complete() {
  local run_id="$1" run_dir="$RESULTS_ROOT/$run_id"
  [[ -f "$run_dir/.done" ]] || return 1
  [[ -f "$run_dir/config.json" ]] || return 1
  grep -Eq '^Round +100 \|' "$run_dir/console.log" 2>/dev/null || return 1
  grep -Fq '"checkpoint_hash": "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f"' \
    "$run_dir/config.json" || return 1
  grep -Fq '"min_require_size": 0' "$run_dir/config.json" || return 1
  grep -Fq '"partition_scheme": "partition-nonempty-v4"' \
    "$run_dir/config.json" || return 1
}

legacy_run_dir() {
  local dataset="$1" alpha="$2" total_clients="$3"
  local training_seed="$4" method="$5"
  [[ "$dataset" == cifar100 || "$dataset" == tinyimagenet ]] || return 1
  [[ "$alpha" == 0.01 && "$total_clients" == 10 ]] || return 1
  [[ "$method" == fedproref || "$method" == proto_aug ]] || return 1
  printf '%s/mcstress_%s_a001_ps42_ts%s_%s\n' \
    "$LEGACY_RESULTS_ROOT" "$dataset" "$training_seed" "$method"
}

legacy_complete() {
  local dataset="$1" alpha="$2" total_clients="$3"
  local select_clients="$4" partition_seed="$5" training_seed="$6" method="$7"
  local run_dir command_file partition_report
  [[ "$select_clients" == 10 && "$partition_seed" == 42 ]] || return 1
  run_dir="$(legacy_run_dir "$dataset" "$alpha" "$total_clients" "$training_seed" "$method")" \
    || return 1
  command_file="$run_dir/command.sh"
  partition_report="$LEGACY_RESULTS_ROOT/partitions/${dataset}_a001_ps42.json"

  [[ -f "$run_dir/.done" && -f "$command_file" ]] || return 1
  grep -Eq '^Round +100 \|' "$run_dir/console.log" 2>/dev/null || return 1
  grep -q 'Best: Round' "$run_dir/console.log" || return 1
  grep -Fq -- '--backbone ViT-B-16' "$command_file" || return 1
  grep -Fq -- '--pretrained ' "$command_file" || return 1
  grep -Fq -- 'old_open_clip_model.safetensors' "$command_file" || return 1
  grep -Fq -- '--comm_rounds 100' "$command_file" || return 1
  grep -Fq -- '--local_epochs 10' "$command_file" || return 1
  grep -Fq -- "--method $method" "$command_file" || return 1
  grep -Fq -- "--seed $training_seed" "$command_file" || return 1
  grep -Fq -- '--partition_seed 42' "$command_file" || return 1
  grep -Fq -- '--num_clients 10' "$command_file" || return 1
  grep -Fq -- '--select_clients 10' "$command_file" || return 1
  grep -Eq '"zero_pairs": [1-9][0-9]*' "$partition_report" 2>/dev/null || return 1
}

completion_source() {
  local run_id="$1" dataset="$2" alpha="$3" total_clients="$4"
  local select_clients="$5" partition_seed="$6" training_seed="$7" method="$8"
  if current_complete "$run_id"; then
    printf 'current\n'
    return 0
  fi
  if legacy_complete "$dataset" "$alpha" "$total_clients" "$select_clients" \
      "$partition_seed" "$training_seed" "$method"; then
    printf 'legacy\n'
    return 0
  fi
  return 1
}

print_missing() {
  local run_id dataset alpha total_clients select_clients
  local partition_seed training_seed method
  while IFS=$'\t' read -r run_id dataset alpha total_clients select_clients \
      partition_seed training_seed method; do
    if ! completion_source "$run_id" "$dataset" "$alpha" "$total_clients" \
        "$select_clients" "$partition_seed" "$training_seed" "$method" >/dev/null; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$run_id" "$dataset" "$alpha" "$total_clients" "$select_clients" \
        "$partition_seed" "$training_seed" "$method"
    fi
  done < <(matrix)
}

print_status() {
  local run_id dataset alpha total_clients select_clients source
  local partition_seed training_seed method
  local total=0 current=0 legacy=0 missing=0
  while IFS=$'\t' read -r run_id dataset alpha total_clients select_clients \
      partition_seed training_seed method; do
    total=$((total + 1))
    if source="$(completion_source "$run_id" "$dataset" "$alpha" "$total_clients" \
        "$select_clients" "$partition_seed" "$training_seed" "$method")"; then
      if [[ "$source" == current ]]; then
        current=$((current + 1))
      else
        legacy=$((legacy + 1))
      fi
    else
      missing=$((missing + 1))
    fi
  done < <(matrix)
  printf 'total=%s\ncomplete_current=%s\ncomplete_legacy=%s\nmissing=%s\n' \
    "$total" "$current" "$legacy" "$missing"
}

preflight() {
  [[ "$MAX_JOBS" =~ ^[1-3]$ ]] || {
    echo "MAX_JOBS must be 1, 2, or 3; got: $MAX_JOBS" >&2
    return 2
  }
  [[ -f "$PROJECT/federated_loop.py" ]] || {
    echo "FedProRef entry point not found: $PROJECT/federated_loop.py" >&2
    return 1
  }
  [[ -f "$VIT_PRETRAINED" ]] || {
    echo "Local ViT-B/16 checkpoint not found: $VIT_PRETRAINED" >&2
    return 1
  }

  local checksum_line checkpoint_hash
  checksum_line="$(sha256sum -- "$VIT_PRETRAINED")"
  checkpoint_hash="${checksum_line%% *}"
  [[ "$checkpoint_hash" == "$VIT_PRETRAINED_SHA256" ]] || {
    echo "ViT-B/16 checkpoint checksum mismatch: $checkpoint_hash" >&2
    return 1
  }

  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)
  "${prefix[@]}" -c \
    'import numpy, open_clip, torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
  mkdir -p "$DATA_ROOT" "$RESULTS_ROOT"
}

write_config() {
  local run_dir="$1" run_id="$2" dataset="$3" alpha="$4"
  local total_clients="$5" select_clients="$6" partition_seed="$7"
  local training_seed="$8" method="$9" refiner_type=none
  [[ "$method" == fedproref ]] && refiner_type=mlp
  printf '{\n  "run_id": "%s",\n  "dataset": "%s",\n  "alpha": %s,\n  "min_require_size": 0,\n  "partition_scheme": "partition-nonempty-v4",\n  "num_clients": %s,\n  "select_clients": %s,\n  "partition_seed": %s,\n  "training_seed": %s,\n  "method": "%s",\n  "backbone": "ViT-B-16",\n  "pretrained": "%s",\n  "checkpoint_hash": "%s",\n  "feature_dim": 512,\n  "refiner_type": "%s",\n  "comm_rounds": 100,\n  "local_epochs": 10,\n  "batch_size": 64,\n  "learning_rate": 0.001\n}\n' \
    "$run_id" "$dataset" "$alpha" "$total_clients" "$select_clients" \
    "$partition_seed" "$training_seed" "$method" "$VIT_PRETRAINED" \
    "$VIT_PRETRAINED_SHA256" "$refiner_type" >"$run_dir/config.json"
}

run_one() {
  local run_id="$1" dataset="$2" alpha="$3" total_clients="$4"
  local select_clients="$5" partition_seed="$6" training_seed="$7" method="$8"
  local run_dir="$RESULTS_ROOT/$run_id" source
  if source="$(completion_source "$run_id" "$dataset" "$alpha" "$total_clients" \
      "$select_clients" "$partition_seed" "$training_seed" "$method")"; then
    printf '[SKIP:%s] %s\n' "$source" "$run_id"
    return 0
  fi

  mkdir -p "$run_dir/logs" "$run_dir/checkpoints"
  build_command "$run_id" "$dataset" "$alpha" "$total_clients" \
    "$select_clients" "$partition_seed" "$training_seed" "$method"
  write_config "$run_dir" "$run_id" "$dataset" "$alpha" "$total_clients" \
    "$select_clients" "$partition_seed" "$training_seed" "$method"
  printf '%q ' "${COMMAND[@]}" >"$run_dir/command.sh"
  printf '\n' >>"$run_dir/command.sh"

  printf '[RUN] %s\n' "$run_id"
  set +e
  (cd "$PROJECT" && "${COMMAND[@]}") 2>&1 | tee "$run_dir/console.log"
  local training_status=${PIPESTATUS[0]}
  set -e
  if [[ "$training_status" -ne 0 ]]; then
    echo "[FAIL] $run_id (exit=$training_status)" >&2
    return "$training_status"
  fi
  if ! grep -Eq '^Round +100 \|' "$run_dir/console.log"; then
    echo "[FAIL] $run_id has no completed Round 100 record" >&2
    return 1
  fi
  touch "$run_dir/.done"
  echo "[DONE] $run_id"
}

run_all() {
  local run_id dataset alpha total_clients select_clients
  local partition_seed training_seed method
  local running=0 failed=0
  preflight
  while IFS=$'\t' read -r run_id dataset alpha total_clients select_clients \
      partition_seed training_seed method; do
    while ((running >= MAX_JOBS)); do
      if ! wait -n; then
        failed=1
      fi
      running=$((running - 1))
      ((failed == 0)) || break
    done
    ((failed == 0)) || break

    run_one "$run_id" "$dataset" "$alpha" "$total_clients" \
      "$select_clients" "$partition_seed" "$training_seed" "$method" &
    running=$((running + 1))
  done < <(matrix)

  while ((running > 0)); do
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  done
  print_status
  ((failed == 0))
}

action="${1:-}"
case "$action" in
  plan)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    matrix
    ;;
  missing)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    print_missing
    ;;
  status)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    print_status
    ;;
  command)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    print_command "$2"
    ;;
  all)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    run_all
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
