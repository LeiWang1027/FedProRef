#!/usr/bin/env bash
set -euo pipefail

PLAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FEDPROREF_PROJECT:-$PLAN_ROOT}"
DATA_ROOT="${FEDPROREF_DATA_ROOT:-$PROJECT/data}"
RESULTS_ROOT="${FEDPROREF_VIT_RESULTS_ROOT:-$PROJECT/results/fedavg_vit_alpha_144}"
CONDA_ENV="${FEDPROREF_CONDA_ENV:-fedfm}"
PYTHON_BIN="${FEDPROREF_PYTHON_BIN:-}"
VIT_PRETRAINED="${FEDPROREF_VIT_PRETRAINED:-$PROJECT/pretrain_path/old_open_clip_model.safetensors}"
VIT_PRETRAINED_SHA256="4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f"

export PYTHONDONTWRITEBYTECODE=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash run_fedavg_vit_alpha_144.sh plan
  bash run_fedavg_vit_alpha_144.sh command RUN_ID
  bash run_fedavg_vit_alpha_144.sh all

The fixed matrix contains 144 sequential FedAvg/ViT-B-16 runs.
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
  local dataset min_require_size alpha training_seed tag run_id
  for dataset in cifar10 cifar100 tinyimagenet; do
    for min_require_size in 0 1; do
      for alpha in 0.01 0.03 0.05 0.07 0.09 0.10 0.30 0.50; do
        tag="$(alpha_tag "$alpha")"
        for training_seed in 42 43 44; do
          run_id="vit_fedavg_${dataset}_${tag}_minpc${min_require_size}_ps42_ts${training_seed}"
          printf '%s\t%s\t%s\t%s\t42\t%s\tfedavg\n' \
            "$run_id" "$dataset" "$alpha" "$min_require_size" "$training_seed"
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
  local run_id="$1" dataset="$2" alpha="$3" min_require_size="$4"
  local partition_seed="$5" training_seed="$6"
  local run_dir="$RESULTS_ROOT/$run_id"
  local -a prefix=()
  while IFS= read -r -d '' token; do prefix+=("$token"); done < <(python_prefix)

  COMMAND=(
    "${prefix[@]}" "$PROJECT/federated_loop.py"
    --dataset "$dataset"
    --data_dir "$DATA_ROOT"
    --num_clients 10
    --select_clients 10
    --alpha "$alpha"
    --min_require_size "$min_require_size"
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
    --refiner_type none
    --method fedavg
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
  local row run_id dataset alpha min_require_size partition_seed training_seed method
  row="$(find_run "$1")"
  IFS=$'\t' read -r run_id dataset alpha min_require_size \
    partition_seed training_seed method <<<"$row"
  build_command "$run_id" "$dataset" "$alpha" "$min_require_size" \
    "$partition_seed" "$training_seed"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
}

preflight() {
  [[ -f "$PROJECT/federated_loop.py" ]] || {
    echo "FedProRef entry point not found: $PROJECT/federated_loop.py" >&2
    return 1
  }
  [[ -f "$VIT_PRETRAINED" ]] || {
    echo "Local ViT checkpoint not found: $VIT_PRETRAINED" >&2
    return 1
  }

  local checksum_line checkpoint_hash
  checksum_line="$(sha256sum -- "$VIT_PRETRAINED")"
  checkpoint_hash="${checksum_line%% *}"
  [[ "$checkpoint_hash" == "$VIT_PRETRAINED_SHA256" ]] || {
    echo "ViT checkpoint checksum mismatch: $checkpoint_hash" >&2
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
  local min_require_size="$5" partition_seed="$6" training_seed="$7"
  printf '{\n  "run_id": "%s",\n  "dataset": "%s",\n  "alpha": %s,\n  "min_require_size": %s,\n  "partition_seed": %s,\n  "training_seed": %s,\n  "method": "fedavg",\n  "backbone": "ViT-B-16",\n  "pretrained": "%s",\n  "checkpoint_hash": "%s",\n  "feature_dim": 512,\n  "num_clients": 10,\n  "select_clients": 10,\n  "refiner_type": "none",\n  "comm_rounds": 100,\n  "local_epochs": 10,\n  "batch_size": 64,\n  "learning_rate": 0.001\n}\n' \
    "$run_id" "$dataset" "$alpha" "$min_require_size" "$partition_seed" \
    "$training_seed" "$VIT_PRETRAINED" "$VIT_PRETRAINED_SHA256" \
    >"$run_dir/config.json"
}

run_one() {
  local run_id="$1" dataset="$2" alpha="$3" min_require_size="$4"
  local partition_seed="$5" training_seed="$6"
  local run_dir="$RESULTS_ROOT/$run_id"

  if [[ -f "$run_dir/.done" ]] && \
     grep -Eq '^Round +100 \|' "$run_dir/console.log" 2>/dev/null; then
    echo "[SKIP] $run_id"
    return 0
  fi

  mkdir -p "$run_dir/logs" "$run_dir/checkpoints"
  build_command "$run_id" "$dataset" "$alpha" "$min_require_size" \
    "$partition_seed" "$training_seed"
  write_config "$run_dir" "$run_id" "$dataset" "$alpha" \
    "$min_require_size" "$partition_seed" "$training_seed"
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
  local run_id dataset alpha min_require_size partition_seed training_seed method
  preflight
  while IFS=$'\t' read -r run_id dataset alpha min_require_size \
    partition_seed training_seed method; do
    run_one "$run_id" "$dataset" "$alpha" "$min_require_size" \
      "$partition_seed" "$training_seed"
  done < <(matrix)
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
