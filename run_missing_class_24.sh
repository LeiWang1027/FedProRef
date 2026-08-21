#!/usr/bin/env bash
set -euo pipefail

PLAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FEDPROREF_PROJECT:-$PLAN_ROOT}"
DATA_ROOT="${FEDPROREF_DATA_ROOT:-$PROJECT/data}"
RESULTS_ROOT="${FEDPROREF_RESULTS_ROOT:-$PLAN_ROOT/results}"
CONDA_ENV="${FEDPROREF_CONDA_ENV:-fedfm}"
PYTHON_BIN="${FEDPROREF_PYTHON_BIN:-}"
PRETRAINED_PATH="${FEDPROREF_PRETRAINED:-$PROJECT/pretrain_path/old_open_clip_model.safetensors}"

export PYTHONDONTWRITEBYTECODE=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash run_missing_class_24.sh list [pilot|cifar100|tinyimagenet|all]
  bash run_missing_class_24.sh command RUN_ID
  bash run_missing_class_24.sh pilot
  bash run_missing_class_24.sh cifar100
  bash run_missing_class_24.sh tinyimagenet
  bash run_missing_class_24.sh all
  bash run_missing_class_24.sh summarize

Recommended sequence:
  1. bash run_missing_class_24.sh pilot       # 2 runs: C100 ps42/ts42
  2. bash run_missing_class_24.sh cifar100    # skips completed pilot; 18 total
  3. bash run_missing_class_24.sh tinyimagenet # 6 runs
  4. bash run_missing_class_24.sh summarize

Environment overrides:
  FEDPROREF_PROJECT, FEDPROREF_DATA_ROOT, FEDPROREF_RESULTS_ROOT,
  FEDPROREF_CONDA_ENV, FEDPROREF_PYTHON_BIN, FEDPROREF_PRETRAINED
EOF
}

emit_run() {
  local dataset="$1" partition_seed="$2" training_seed="$3" method="$4"
  local run_id="mcstress_${dataset}_a001_ps${partition_seed}_ts${training_seed}_${method}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$run_id" "$dataset" "$partition_seed" "$training_seed" "$method"
}

matrix() {
  local target="${1:-all}"
  local partition_seed training_seed method
  case "$target" in
    pilot)
      for method in proto_aug fedproref; do
        emit_run cifar100 42 42 "$method"
      done
      ;;
    cifar100)
      for partition_seed in 42 43 44; do
        for training_seed in 42 43 44; do
          for method in proto_aug fedproref; do
            emit_run cifar100 "$partition_seed" "$training_seed" "$method"
          done
        done
      done
      ;;
    tinyimagenet)
      for training_seed in 42 43 44; do
        for method in proto_aug fedproref; do
          emit_run tinyimagenet 42 "$training_seed" "$method"
        done
      done
      ;;
    all)
      matrix cifar100
      matrix tinyimagenet
      ;;
    *)
      echo "Unknown target: $target" >&2
      return 2
      ;;
  esac
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
  local run_id="$1" dataset="$2" partition_seed="$3" training_seed="$4" method="$5"
  local run_dir="$RESULTS_ROOT/$run_id"
  local refiner_type=none
  [[ "$method" == fedproref ]] && refiner_type=mlp

  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)
  COMMAND=(
    "${prefix[@]}" "$PROJECT/federated_loop.py"
    --dataset "$dataset"
    --data_dir "$DATA_ROOT"
    --num_clients 10
    --select_clients 10
    --alpha 0.01
    --min_samples_per_class 10
    --backbone ViT-B-16
    --pretrained "$PRETRAINED_PATH"
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
    [[ "${row%%$'\t'*}" == "$wanted" ]] && { printf '%s\n' "$row"; return 0; }
  done < <(matrix all)
  echo "Unknown run ID: $wanted" >&2
  return 2
}

print_command() {
  local run_id="$1" row dataset partition_seed training_seed method
  row="$(find_run "$run_id")"
  IFS=$'\t' read -r run_id dataset partition_seed training_seed method <<<"$row"
  build_command "$run_id" "$dataset" "$partition_seed" "$training_seed" "$method"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
}

preflight() {
  [[ -f "$PROJECT/federated_loop.py" ]] || {
    echo "FedProRef entry point not found: $PROJECT/federated_loop.py" >&2
    return 1
  }
  [[ -f "$PROJECT/config.py" ]] || {
    echo "FedProRef config not found: $PROJECT/config.py" >&2
    return 1
  }
  [[ -f "$PRETRAINED_PATH" ]] || {
    echo "Local pretrained checkpoint not found: $PRETRAINED_PATH" >&2
    return 1
  }
  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)
  "${prefix[@]}" -c 'import numpy, open_clip, torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
  mkdir -p "$DATA_ROOT" "$RESULTS_ROOT" "$RESULTS_ROOT/partitions"
}

cache_path() {
  local dataset="$1" partition_seed="$2" pretrained_tag checksum_line checkpoint_hash
  pretrained_tag="$(basename -- "$PRETRAINED_PATH")"
  pretrained_tag="${pretrained_tag// /_}"
  checksum_line="$(sha256sum -- "$PRETRAINED_PATH")"
  checkpoint_hash="${checksum_line%% *}"
  [[ "$checkpoint_hash" =~ ^[0-9a-f]{64}$ ]] || {
    echo "Could not resolve checkpoint SHA-256: $PRETRAINED_PATH" >&2
    return 1
  }
  printf '%s/%s_clip_cache/ViT-B-16_%s_%s_minpc0_partition-nonempty-v4_alpha0.01_c10_ps%s.npz\n' \
    "$DATA_ROOT" "$dataset" "$pretrained_tag" "${checkpoint_hash:0:12}" "$partition_seed"
}

write_partition_report() {
  local dataset="$1" partition_seed="$2" classes cache output
  cache="$(cache_path "$dataset" "$partition_seed")"
  output="$RESULTS_ROOT/partitions/${dataset}_a001_ps${partition_seed}.json"
  [[ -f "$cache" ]] || {
    echo "Expected partition cache not found: $cache" >&2
    return 1
  }
  [[ "$dataset" == cifar100 ]] && classes=100 || classes=200
  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)
  "${prefix[@]}" "$PLAN_ROOT/partition_report.py" \
    --cache "$cache" --output "$output" --dataset "$dataset" \
    --partition-seed "$partition_seed" --num-classes "$classes" \
    --upload-threshold 10 >/dev/null
}

run_one() {
  local run_id="$1" dataset="$2" partition_seed="$3" training_seed="$4" method="$5"
  local run_dir="$RESULTS_ROOT/$run_id"
  if [[ -f "$run_dir/.done" ]] && grep -q 'Best: Round' "$run_dir/console.log" 2>/dev/null; then
    echo "[SKIP] $run_id"
    return 0
  fi

  mkdir -p "$run_dir/logs" "$run_dir/checkpoints"
  build_command "$run_id" "$dataset" "$partition_seed" "$training_seed" "$method"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_id" "$dataset" "$partition_seed" "$training_seed" "$method" "$(date -Is)" \
    >> "$RESULTS_ROOT/manifest.tsv"
  printf '[RUN] %s\n' "$run_id"
  printf '%q ' "${COMMAND[@]}" > "$run_dir/command.sh"
  printf '\n' >> "$run_dir/command.sh"

  set +e
  (cd "$PROJECT" && "${COMMAND[@]}") 2>&1 | tee "$run_dir/console.log"
  local training_status=${PIPESTATUS[0]}
  set -e
  if [[ "$training_status" -ne 0 ]]; then
    echo "[FAIL] $run_id (exit=$training_status)" >&2
    return "$training_status"
  fi
  if ! grep -q 'Best: Round' "$run_dir/console.log"; then
    echo "[FAIL] $run_id completed without a parsable Best line" >&2
    return 1
  fi
  write_partition_report "$dataset" "$partition_seed"
  touch "$run_dir/.done"
  echo "[DONE] $run_id"
}

run_target() {
  local target="$1" run_id dataset partition_seed training_seed method
  preflight
  while IFS=$'\t' read -r run_id dataset partition_seed training_seed method; do
    run_one "$run_id" "$dataset" "$partition_seed" "$training_seed" "$method"
  done < <(matrix "$target")
  summarize
}

summarize() {
  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)
  "${prefix[@]}" "$PLAN_ROOT/summarize_missing_class.py" \
    --results-root "$RESULTS_ROOT"
}

action="${1:-}"
case "$action" in
  list)
    matrix "${2:-all}"
    ;;
  command)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    print_command "$2"
    ;;
  pilot|cifar100|tinyimagenet|all)
    run_target "$action"
    ;;
  summarize)
    summarize
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
