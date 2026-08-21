#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT/run_vit_minpc0_methods_432.sh"
FIXTURE_ROOT="$(mktemp -d /tmp/vit-minpc0-methods-plan-test.XXXXXX)"

cleanup_fixture() {
  local resolved
  resolved="$(realpath -- "$FIXTURE_ROOT")"
  [[ "$resolved" == /tmp/vit-minpc0-methods-plan-test.* ]] || return 1
  [[ -d "$resolved" && ! -L "$resolved" ]] || return 1
  rm -rf -- "$resolved"
}
trap cleanup_fixture EXIT

RESULTS_ROOT="$FIXTURE_ROOT/results"
LEGACY_ROOT="$FIXTURE_ROOT/legacy"
RUN_ID="vit_minpc0_cifar10_a001_c10_sc10_ps42_ts42_fedproref"
RUN_DIR="$RESULTS_ROOT/$RUN_ID"

# Regression: the active plan contains c10 only, with three TinyImageNet alphas.
plan="$(bash "$RUNNER" plan)"
[[ "$(wc -l <<<"$plan")" -eq 171 ]]
awk -F'\t' '$4 != 10 {exit 1}' <<<"$plan"
[[ "$(awk -F'\t' '$2 == "cifar10" {n++} END {print n+0}' <<<"$plan")" -eq 72 ]]
[[ "$(awk -F'\t' '$2 == "cifar100" {n++} END {print n+0}' <<<"$plan")" -eq 72 ]]
[[ "$(awk -F'\t' '$2 == "tinyimagenet" {n++} END {print n+0}' <<<"$plan")" -eq 27 ]]
awk -F'\t' '$2 == "tinyimagenet" && $3 != 0.01 && $3 != 0.03 && $3 != 0.05 {exit 1}' <<<"$plan"

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/.done"
printf '%s\n' 'Round  100 | acc: 1.0000 | ece: 0.0000 | nll: 0.0000' \
  >"$RUN_DIR/console.log"

write_config() {
  local scheme_line="$1"
  printf '{\n  "checkpoint_hash": "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f",\n  "min_require_size": 0%s\n}\n' \
    "$scheme_line" >"$RUN_DIR/config.json"
}

status() {
  FEDPROREF_VIT_MINPC0_RESULTS_ROOT="$RESULTS_ROOT" \
  FEDPROREF_LEGACY_RESULTS_ROOT="$LEGACY_ROOT" \
    bash "$RUNNER" status
}

write_config ''
without_scheme="$(status)"
grep -Fxq 'complete_current=0' <<<"$without_scheme"
grep -Fxq 'missing=171' <<<"$without_scheme"

write_config ',\n  "partition_scheme": "partition-nonempty-v4"'
with_scheme="$(status)"
grep -Fxq 'complete_current=1' <<<"$with_scheme"
grep -Fxq 'missing=170' <<<"$with_scheme"

echo "ViT minpc0 method plan partition-version contract: PASS"
