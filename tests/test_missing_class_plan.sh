#!/usr/bin/env bash
set -euo pipefail

PLAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PLAN_ROOT/run_missing_class_24.sh"

all_rows="$(bash "$RUNNER" list all)"
pilot_rows="$(bash "$RUNNER" list pilot)"

[[ "$(printf '%s\n' "$all_rows" | awk 'NF {n++} END {print n+0}')" -eq 24 ]]
[[ "$(printf '%s\n' "$all_rows" | awk -F '\t' '$2=="cifar100" {n++} END {print n+0}')" -eq 18 ]]
[[ "$(printf '%s\n' "$all_rows" | awk -F '\t' '$2=="tinyimagenet" {n++} END {print n+0}')" -eq 6 ]]
[[ "$(printf '%s\n' "$pilot_rows" | awk 'NF {n++} END {print n+0}')" -eq 2 ]]

methods="$(printf '%s\n' "$all_rows" | cut -f5 | sort -u | paste -sd, -)"
[[ "$methods" == "fedproref,proto_aug" ]]

if grep -Fq $'\tfedavg' <<<"$all_rows"; then
  echo "FedAvg must not appear in the focused experiment matrix" >&2
  exit 1
fi

[[ "$(printf '%s\n' "$all_rows" | awk -F '\t' '$2=="tinyimagenet" {print $3}' | sort -u | paste -sd, -)" == "42" ]]
[[ "$(printf '%s\n' "$all_rows" | awk -F '\t' '$2=="cifar100" {print $3}' | sort -u | paste -sd, -)" == "42,43,44" ]]
[[ "$(printf '%s\n' "$all_rows" | awk -F '\t' '{print $4}' | sort -u | paste -sd, -)" == "42,43,44" ]]

fedproref_cmd="$(bash "$RUNNER" command mcstress_cifar100_a001_ps42_ts42_fedproref)"
proto_aug_cmd="$(bash "$RUNNER" command mcstress_cifar100_a001_ps42_ts42_proto_aug)"

grep -Fq -- 'env -u ALL_PROXY -u all_proxy' <<<"$fedproref_cmd"
grep -Fq -- '--method fedproref' <<<"$fedproref_cmd"
grep -Fq -- '--refiner_type mlp' <<<"$fedproref_cmd"
grep -Fq -- '--method proto_aug' <<<"$proto_aug_cmd"
grep -Fq -- '--refiner_type none' <<<"$proto_aug_cmd"

for fixed_arg in \
  "--data_dir $PLAN_ROOT/data" \
  '--alpha 0.01' '--num_clients 10' '--select_clients 10' \
  '--comm_rounds 100' '--local_epochs 10' '--batch_size 64' \
  '--backbone ViT-B-16' \
  "--pretrained $PLAN_ROOT/pretrain_path/old_open_clip_model.safetensors" \
  '--num_modes 1' \
  '--min_samples_per_class 10' '--aug_gen_per_class 100' \
  '--weak_class_percentile 10.0'; do
  grep -Fq -- "$fixed_arg" <<<"$fedproref_cmd"
done

if grep -Fq -- '--pretrained openai' <<<"$fedproref_cmd"; then
  echo "The focused experiment must use the local pretrained checkpoint" >&2
  exit 1
fi

if grep -Eq -- '--augmentation_scope|--min_client_samples|direct_anchor_aug|proto_aug_all|fedproref_all|fedproref_weak_only' <<<"$fedproref_cmd"; then
  echo "Unsupported method or option found in command" >&2
  exit 1
fi

# Regression: a local-checkpoint cache must be found after training so the
# runner marks the run complete and advances to the next matrix row.
fixture_root="$(mktemp -d /tmp/fedproref-missing-class-test.XXXXXX)"
cleanup_fixture() {
  local resolved
  resolved="$(realpath -- "$fixture_root")"
  [[ "$resolved" == /tmp/fedproref-missing-class-test.* ]] || return 1
  [[ -d "$resolved" && ! -L "$resolved" ]] || return 1
  rm -rf -- "$resolved"
}
trap cleanup_fixture EXIT
fixture_project="$fixture_root/project"
fixture_data="$fixture_root/data"
fixture_results="$fixture_root/results"
mkdir -p \
  "$fixture_project/pretrain_path" \
  "$fixture_data/cifar100_clip_cache"
touch \
  "$fixture_project/federated_loop.py" \
  "$fixture_project/config.py" \
  "$fixture_project/pretrain_path/old_open_clip_model.safetensors" \
  "$fixture_data/cifar100_clip_cache/ViT-B-16_old_open_clip_model.safetensors_e3b0c44298fc_minpc0_partition-nonempty-v4_alpha0.01_c10_ps42.npz"

fake_python="$fixture_root/fake-python"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "${1:-}" in' \
  '  */federated_loop.py) printf "%s\\n" "Best: Round 1 | Acc=1.00%" ;;' \
  'esac' >"$fake_python"
chmod +x "$fake_python"

FEDPROREF_PROJECT="$fixture_project" \
FEDPROREF_DATA_ROOT="$fixture_data" \
FEDPROREF_RESULTS_ROOT="$fixture_results" \
FEDPROREF_PYTHON_BIN="$fake_python" \
  bash "$RUNNER" pilot >/dev/null

[[ -f "$fixture_results/mcstress_cifar100_a001_ps42_ts42_proto_aug/.done" ]]
[[ -f "$fixture_results/mcstress_cifar100_a001_ps42_ts42_fedproref/.done" ]]

echo "missing-class plan contract: PASS"
