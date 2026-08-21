#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT/run_fedavg_vit_alpha_144.sh"

fixture_root="$(mktemp -d /tmp/fedavg-vit-alpha-plan-test.XXXXXX)"
cleanup_fixture() {
  local resolved
  resolved="$(realpath -- "$fixture_root")"
  [[ "$resolved" == /tmp/fedavg-vit-alpha-plan-test.* ]] || return 1
  [[ -d "$resolved" && ! -L "$resolved" ]] || return 1
  rm -rf -- "$resolved"
}
trap cleanup_fixture EXIT

readonly_results="$fixture_root/readonly-results"
plan="$(FEDPROREF_VIT_RESULTS_ROOT="$readonly_results" bash "$RUNNER" plan)"
[[ "$(awk 'NF {n++} END {print n+0}' <<<"$plan")" -eq 144 ]]
[[ "$(cut -f1 <<<"$plan" | sort -u | wc -l)" -eq 144 ]]
[[ "$(cut -f2 <<<"$plan" | sort -u | paste -sd, -)" == "cifar10,cifar100,tinyimagenet" ]]
[[ "$(cut -f3 <<<"$plan" | sort -u | paste -sd, -)" == "0.01,0.03,0.05,0.07,0.09,0.10,0.30,0.50" ]]
[[ "$(cut -f4 <<<"$plan" | sort -u | paste -sd, -)" == "0,1" ]]
[[ "$(cut -f5 <<<"$plan" | sort -u)" == "42" ]]
[[ "$(cut -f6 <<<"$plan" | sort -u | paste -sd, -)" == "42,43,44" ]]
[[ "$(cut -f7 <<<"$plan" | sort -u)" == "fedavg" ]]

first_id="$(head -1 <<<"$plan" | cut -f1)"
last_id="$(tail -1 <<<"$plan" | cut -f1)"
[[ "$first_id" == "vit_fedavg_cifar10_a001_minpc0_ps42_ts42" ]]
[[ "$last_id" == "vit_fedavg_tinyimagenet_a050_minpc1_ps42_ts44" ]]

representative="vit_fedavg_cifar100_a010_minpc1_ps42_ts43"
command="$(FEDPROREF_VIT_RESULTS_ROOT="$readonly_results" \
  bash "$RUNNER" command "$representative")"
for expected in \
  '--dataset cifar100' '--alpha 0.10' '--min_require_size 1' \
  '--partition_seed 42' '--seed 43' '--method fedavg' \
  '--backbone ViT-B-16' '--feat_dim 512' '--refiner_type none' \
  '--num_clients 10' '--select_clients 10' '--comm_rounds 100' \
  '--local_epochs 10' '--batch_size 64' '--lr 0.001'; do
  grep -Fq -- "$expected" <<<"$command"
done
grep -Fq -- \
  "--pretrained $PROJECT/pretrain_path/old_open_clip_model.safetensors" \
  <<<"$command"

if bash "$RUNNER" command not_a_real_run >/dev/null 2>&1; then
  echo "unknown run ID unexpectedly succeeded" >&2
  exit 1
fi

[[ ! -e "$readonly_results" ]]

fake_python="$fixture_root/fake-python"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'if [[ "${1:-}" == "-c" ]]; then exit 0; fi' \
  'printf "%s\n" "Round  100 | acc: 1.0000 | ece: 0.0000 | nll: 0.0000 | round_time: 0.0001"' \
  'printf "%s\n" "Best: Round 100 | Acc=1.0000%"' \
  >"$fake_python"
chmod +x "$fake_python"

fake_checkpoint="$fixture_root/old_open_clip_model.safetensors"
fake_bin="$fixture_root/bin"
touch "$fake_checkpoint"
mkdir -p "$fake_bin"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'printf "%s  %s\\n" "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f" "${1:-checkpoint}"' \
  >"$fake_bin/sha256sum"
chmod +x "$fake_bin/sha256sum"

fake_results="$fixture_root/results"
PATH="$fake_bin:$PATH" \
FEDPROREF_VIT_PRETRAINED="$fake_checkpoint" \
FEDPROREF_VIT_RESULTS_ROOT="$fake_results" \
FEDPROREF_PYTHON_BIN="$fake_python" \
  bash "$RUNNER" all >/dev/null

[[ "$(find "$fake_results" -mindepth 2 -maxdepth 2 -name .done -type f | wc -l)" -eq 144 ]]
[[ "$(find "$fake_results" -mindepth 2 -maxdepth 2 -name config.json -type f | wc -l)" -eq 144 ]]

second_output="$(PATH="$fake_bin:$PATH" \
  FEDPROREF_VIT_PRETRAINED="$fake_checkpoint" \
  FEDPROREF_VIT_RESULTS_ROOT="$fake_results" \
  FEDPROREF_PYTHON_BIN="$fake_python" bash "$RUNNER" all)"
[[ "$(grep -c '^\[SKIP\]' <<<"$second_output")" -eq 144 ]]

echo "FedAvg ViT alpha plan and resume contract: PASS"
