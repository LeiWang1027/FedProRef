#!/usr/bin/env bash
set -euo pipefail

PLAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FEDPROREF_PROJECT:-$PLAN_ROOT}"
DATA_ROOT="${FEDPROREF_DATA_ROOT:-$PROJECT/data}"
RESULTS_ROOT="${FEDPROREF_RN50_RESULTS_ROOT:-$PROJECT/results/backbone_rn50_48}"
RN50_PRETRAINED="${FEDPROREF_RN50_PRETRAINED:-$PROJECT/pretrain_path/RN50_openai.pt}"
CONDA_ENV="${FEDPROREF_CONDA_ENV:-fedfm}"
PYTHON_BIN="${FEDPROREF_PYTHON_BIN:-}"

export PYTHONDONTWRITEBYTECODE=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash run_backbone_rn50_48.sh plan
  bash run_backbone_rn50_48.sh command RUN_ID
  bash run_backbone_rn50_48.sh all
  bash run_backbone_rn50_48.sh summarize

The full matrix contains 48 sequential, resumable RN50/OpenAI runs.
EOF
}

emit_run() {
  local dataset="$1" min_require_size="$2" training_seed="$3" method="$4"
  local run_id="rn50_${dataset}_a001_minpc${min_require_size}_ps42_ts${training_seed}_${method}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_id" "$dataset" "$min_require_size" 42 "$training_seed" "$method"
}

matrix() {
  local dataset min_require_size training_seed method
  for dataset in cifar100 tinyimagenet; do
    for min_require_size in 0 1; do
      for training_seed in 42 43 44; do
        for method in fedavg proto_aug direct_anchor_aug fedproref; do
          emit_run "$dataset" "$min_require_size" "$training_seed" "$method"
        done
      done
    done
  done
}

python_prefix() {
  if [[ -n "$PYTHON_BIN" ]]; then
    printf '%s\0' "$PYTHON_BIN"
  else
    printf '%s\0' conda run --no-capture-output -n "$CONDA_ENV" python
  fi
}

build_command() {
  local run_id="$1" dataset="$2" min_require_size="$3"
  local partition_seed="$4" training_seed="$5" method="$6"
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
    --min_require_size "$min_require_size"
    --min_samples_per_class 10
    --backbone RN50
    --pretrained "$RN50_PRETRAINED"
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
  local row run_id dataset min_require_size partition_seed training_seed method
  row="$(find_run "$1")"
  IFS=$'\t' read -r run_id dataset min_require_size partition_seed training_seed method <<<"$row"
  build_command "$run_id" "$dataset" "$min_require_size" \
    "$partition_seed" "$training_seed" "$method"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
}

preflight() {
  [[ -f "$PROJECT/federated_loop.py" ]] || {
    echo "FedProRef entry point not found: $PROJECT/federated_loop.py" >&2
    return 1
  }
  [[ -f "$RN50_PRETRAINED" ]] || {
    echo "Local RN50/OpenAI checkpoint not found: $RN50_PRETRAINED" >&2
    return 1
  }
  local checksum_line checkpoint_hash
  checksum_line="$(sha256sum -- "$RN50_PRETRAINED")"
  checkpoint_hash="${checksum_line%% *}"
  [[ "$checkpoint_hash" == "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762" ]] || {
    echo "RN50/OpenAI checkpoint checksum mismatch: $checkpoint_hash" >&2
    return 1
  }
  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)
  "${prefix[@]}" -c 'import open_clip, torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
  mkdir -p "$DATA_ROOT" "$RESULTS_ROOT"
}

write_config() {
  local run_dir="$1" run_id="$2" dataset="$3" min_require_size="$4"
  local partition_seed="$5" training_seed="$6" method="$7"
  local refiner_type=none
  [[ "$method" == fedproref ]] && refiner_type=mlp
  printf '{\n  "run_id": "%s",\n  "dataset": "%s",\n  "backbone": "RN50",\n  "pretrained": "%s",\n  "pretrained_tag": "openai",\n  "checkpoint_hash": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",\n  "alpha": 0.01,\n  "min_require_size": %s,\n  "partition_seed": %s,\n  "training_seed": %s,\n  "method": "%s",\n  "refiner_type": "%s",\n  "comm_rounds": 100,\n  "local_epochs": 10\n}\n' \
    "$run_id" "$dataset" "$RN50_PRETRAINED" "$min_require_size" "$partition_seed" \
    "$training_seed" "$method" "$refiner_type" > "$run_dir/config.json"
}

run_one() {
  local run_id="$1" dataset="$2" min_require_size="$3"
  local partition_seed="$4" training_seed="$5" method="$6"
  local run_dir="$RESULTS_ROOT/$run_id"
  if [[ -f "$run_dir/.done" ]] && \
     grep -Eq '^Round +100 \|' "$run_dir/console.log" 2>/dev/null; then
    echo "[SKIP] $run_id"
    return 0
  fi

  mkdir -p "$run_dir/logs" "$run_dir/checkpoints"
  build_command "$run_id" "$dataset" "$min_require_size" \
    "$partition_seed" "$training_seed" "$method"
  write_config "$run_dir" "$run_id" "$dataset" "$min_require_size" \
    "$partition_seed" "$training_seed" "$method"
  printf '%q ' "${COMMAND[@]}" > "$run_dir/command.sh"
  printf '\n' >> "$run_dir/command.sh"

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
  local run_id dataset min_require_size partition_seed training_seed method
  preflight
  while IFS=$'\t' read -r run_id dataset min_require_size partition_seed training_seed method; do
    run_one "$run_id" "$dataset" "$min_require_size" \
      "$partition_seed" "$training_seed" "$method"
  done < <(matrix)
  summarize
}

summarize() {
  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)
  "${prefix[@]}" "$PROJECT/summarize_backbone_rn50.py" \
    --results-dir "$RESULTS_ROOT" --output-dir "$RESULTS_ROOT/summary"
}

action="${1:-}"
case "$action" in
  plan)
    matrix
    ;;
  command)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    print_command "$2"
    ;;
  all)
    run_all
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
