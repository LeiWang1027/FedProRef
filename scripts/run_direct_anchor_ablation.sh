#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT_DIR/results/direct_anchor_aug_experiment_manifest.csv"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN=0
RESUME=0
MAX_PARALLEL=2
ONLY_DATASET=""
ONLY_ALPHA=""
ONLY_SEED=""
PREPARE_ONLY=0

usage() {
  echo "Usage: $0 [--dry-run] [--prepare-only] [--resume] [--max-parallel N] [--only-dataset DATASET] [--only-alpha ALPHA] [--only-seed SEED]"
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    --resume) RESUME=1 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift ;;
    --only-dataset) ONLY_DATASET="$2"; shift ;;
    --only-alpha) ONLY_ALPHA="$2"; shift ;;
    --only-seed) ONLY_SEED="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "--max-parallel must be positive" >&2; exit 2; }
cd "$ROOT_DIR"

validate_inputs() {
  local dataset="$1" alpha="$2" cache_dir="data/${dataset}_clip_cache"
  local cache_match partition_match
  # Training uses a partition-independent global feature cache. Older
  # alpha-specific combined caches are accepted for backwards compatibility.
  cache_match="$(find "$cache_dir" -maxdepth 1 -type f \( -name '*_global_features.npz' -o -name "*alpha${alpha}_c10_s42.npz" \) -print -quit 2>/dev/null || true)"
  partition_match="$(find "$cache_dir/partitions" -maxdepth 1 -type f -name "${dataset}_alpha${alpha}_c10_*_ps42.npz" -print -quit 2>/dev/null || true)"
  [[ -n "$cache_match" ]] || { echo "Missing frozen feature cache for $dataset alpha=$alpha" >&2; return 1; }
  if [[ -z "$partition_match" ]]; then
    if ((DRY_RUN)); then
      echo "[PLAN PARTITION REQUIRED] $dataset alpha=$alpha seed=42" >&2
      return 0
    fi
    echo "Missing partition cache for $dataset alpha=$alpha seed=42" >&2
    return 1
  fi
}

prepare_partitions() {
  local dataset alpha partition_seed seed unused key
  declare -A prepared=()
  while IFS=, read -r dataset alpha partition_seed seed unused; do
    [[ "$dataset" == dataset ]] && continue
    [[ -n "$ONLY_DATASET" && "$dataset" != "$ONLY_DATASET" ]] && continue
    [[ -n "$ONLY_ALPHA" && "$alpha" != "$ONLY_ALPHA" ]] && continue
    [[ -n "$ONLY_SEED" && "$seed" != "$ONLY_SEED" ]] && continue
    key="${dataset},${alpha},${partition_seed}"
    [[ -n "${prepared[$key]:-}" ]] && continue
    prepared["$key"]=1
    local cmd=("$PYTHON_BIN" scripts/prepare_direct_anchor_partitions.py --dataset "$dataset" --alpha "$alpha" --partition-seed "$partition_seed" --num-clients 10 --min-require-size 1 --data-dir data)
    ((DRY_RUN)) && cmd+=(--dry-run)
    printf '[PREPARE]'; printf ' %q' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
  done < "$MANIFEST"
}

write_run_files() {
  local run_dir="$1" dataset="$2" alpha="$3" seed="$4"
  "$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); d.mkdir(parents=True,exist_ok=True); (d/"config.json").write_text(json.dumps({"method":"direct_anchor_aug","display_name":"DirectAnchorAug","dataset":sys.argv[2],"alpha":float(sys.argv[3]),"partition_seed":42,"training_seed":int(sys.argv[4]),"total_clients":10,"selected_clients":10,"rounds":100,"local_epochs":10,"prototype_merge":True,"num_aug_per_class":100,"n_min":10,"tau":10,"q":0.10},indent=2)+"\n")' "$run_dir" "$dataset" "$alpha" "$seed"
}

run_one() {
  local dataset="$1" alpha="$2" seed="$3"
  local run_dir="results/direct_anchor_aug/${dataset}/alpha_${alpha}/seed_${seed}"
  local checkpoint="$run_dir/direct_anchor_aug_${dataset}_a${alpha}_s${seed}_direct_anchor_aug.pth"
  local has_files=0
  if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    has_files=1
  fi
  if [[ -f "$checkpoint" ]]; then
    if "$PYTHON_BIN" scripts/finalize_direct_anchor_run.py "$run_dir" --rounds 100; then
      echo "[SKIP COMPLETED] $run_dir"
      return 0
    fi
    echo "[CHECKPOINT WITHOUT COMPLETE LOG] $run_dir" >&2
  fi
  if ((has_files)); then
    if ((RESUME == 0)); then
      echo "[INCOMPLETE] $run_dir (use --resume only after reviewing its logs)" >&2
      return 1
    fi
    echo "[RESTART INCOMPLETE] $run_dir"
  fi
  validate_inputs "$dataset" "$alpha"
  local cmd=("$PYTHON_BIN" federated_loop.py --dataset "$dataset" --data_dir ./data --num_clients 10 --select_clients 10 --alpha "$alpha" --min_require_size 1 --min_samples_per_class 10 --backbone ViT-B-16 --pretrained ./pretrain_path/old_open_clip_model.safetensors --feat_dim 512 --head_type linear --comm_rounds 100 --local_epochs 10 --batch_size 64 --lr 0.001 --num_modes 1 --proto_merge_threshold 0.90 --refiner_type none --proposal_sigma 0.05 --gen_per_class 500 --aug_gen_per_class 100 --weak_class_percentile 10.0 --method direct_anchor_aug --seed "$seed" --partition_seed 42 --device auto --log_dir "$run_dir" --save_dir "$run_dir" --exp_name "direct_anchor_aug_${dataset}_a${alpha}_s${seed}")
  printf '[RUN]'; printf ' %q' "${cmd[@]}"; printf '\n'
  ((DRY_RUN)) && return 0
  local attempt_id
  attempt_id="$(date +%Y%m%d_%H%M%S)"
  local attempt_log="$run_dir/stdout_attempt_${attempt_id}.log"
  if [[ -f "$run_dir/stdout.log" ]]; then
    mv "$run_dir/stdout.log" "$run_dir/stdout_interrupted_${attempt_id}.log"
  fi
  write_run_files "$run_dir" "$dataset" "$alpha" "$seed"
  "${cmd[@]}" 2>&1 | tee "$run_dir/stdout.log" "$attempt_log"
  "$PYTHON_BIN" scripts/finalize_direct_anchor_run.py "$run_dir" --rounds 100
}

# Partition preparation is serial: workers never contend for the same cache
# lock. It reads only local frozen-feature labels and never downloads data.
prepare_partitions
if ((PREPARE_ONLY)); then
  exit 0
fi

running=0
failures=0
while IFS=, read -r dataset alpha partition_seed seed _; do
  [[ "$dataset" == dataset ]] && continue
  [[ -n "$ONLY_DATASET" && "$dataset" != "$ONLY_DATASET" ]] && continue
  [[ -n "$ONLY_ALPHA" && "$alpha" != "$ONLY_ALPHA" ]] && continue
  [[ -n "$ONLY_SEED" && "$seed" != "$ONLY_SEED" ]] && continue
  if ((DRY_RUN || MAX_PARALLEL == 1)); then run_one "$dataset" "$alpha" "$seed" || failures=$((failures+1)); continue; fi
  run_one "$dataset" "$alpha" "$seed" &
  running=$((running+1))
  if ((running >= MAX_PARALLEL)); then wait -n || failures=$((failures+1)); running=$((running-1)); fi
done < "$MANIFEST"
while ((running)); do wait -n || failures=$((failures+1)); running=$((running-1)); done
((failures == 0)) || exit 1
