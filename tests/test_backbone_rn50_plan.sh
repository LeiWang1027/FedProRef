#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT/run_backbone_rn50_48.sh"

plan="$(bash "$RUNNER" plan)"
[[ "$(sed '/^$/d' <<<"$plan" | wc -l)" -eq 48 ]]
[[ "$(cut -f1 <<<"$plan" | sort -u | wc -l)" -eq 48 ]]
[[ "$(cut -f2 <<<"$plan" | sort -u | paste -sd, -)" == "cifar100,tinyimagenet" ]]
[[ "$(cut -f3 <<<"$plan" | sort -u | paste -sd, -)" == "0,1" ]]
[[ "$(cut -f5 <<<"$plan" | sort -u | paste -sd, -)" == "42,43,44" ]]
[[ "$(cut -f6 <<<"$plan" | sort -u | paste -sd, -)" == "direct_anchor_aug,fedavg,fedproref,proto_aug" ]]

fedproref_id="rn50_cifar100_a001_minpc1_ps42_ts42_fedproref"
fedproref_cmd="$(bash "$RUNNER" command "$fedproref_id")"
for expected in \
  '--dataset cifar100' \
  '--backbone RN50' \
  "--pretrained $PROJECT/pretrain_path/RN50_openai.pt" \
  '--alpha 0.01' \
  '--num_clients 10' \
  '--select_clients 10' \
  '--min_require_size 1' \
  '--partition_seed 42' \
  '--seed 42' \
  '--comm_rounds 100' \
  '--local_epochs 10' \
  '--batch_size 64' \
  '--method fedproref' \
  '--refiner_type mlp'; do
  grep -Fq -- "$expected" <<<"$fedproref_cmd"
done
if grep -Fq -- '--feat_dim' <<<"$fedproref_cmd"; then
  echo "RN50 command must infer feature dimension" >&2
  exit 1
fi

# The regular Python entry point must expose the same native controls.
grep -q '"--backbone"' "$PROJECT/run.py"
grep -q '"--min_require_size"' "$PROJECT/run.py"
grep -q '"direct_anchor_aug"' "$PROJECT/run.py"
grep -q 'RN50_openai.pt' "$PROJECT/run.py"
if grep -q '^BACKBONE[[:space:]]*=[[:space:]]*"ViT-B-16"' "$PROJECT/run.py"; then
  echo "run.py still hard-codes ViT-B-16" >&2
  exit 1
fi
if grep -q '^FEAT_DIM[[:space:]]*=[[:space:]]*512' "$PROJECT/run.py"; then
  echo "run.py still hard-codes a 512-dimensional feature space" >&2
  exit 1
fi
grep -Fq -- 'shift 3' "$PROJECT/run_experiment.sh"
grep -Fq -- '"$@"' "$PROJECT/run_experiment.sh"

fedavg_id="rn50_tinyimagenet_a001_minpc0_ps42_ts44_fedavg"
fedavg_cmd="$(bash "$RUNNER" command "$fedavg_id")"
grep -Fq -- '--dataset tinyimagenet' <<<"$fedavg_cmd"
grep -Fq -- '--min_require_size 0' <<<"$fedavg_cmd"
grep -Fq -- '--seed 44' <<<"$fedavg_cmd"
grep -Fq -- '--method fedavg' <<<"$fedavg_cmd"
grep -Fq -- '--refiner_type none' <<<"$fedavg_cmd"

if bash "$RUNNER" command not_a_real_run >/dev/null 2>&1; then
  echo "unknown run ID unexpectedly succeeded" >&2
  exit 1
fi

echo "RN50 48-run plan test passed"
