#!/usr/bin/env bash
set -euo pipefail

if ! command -v flock >/dev/null 2>&1; then
  echo "ViT minpc0 parallel scheduling test: SKIP (flock unavailable)"
  exit 0
fi

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT/run_vit_minpc0_methods_432.sh"
FIXTURE_ROOT="$(mktemp -d /tmp/vit-minpc0-methods-parallel-test.XXXXXX)"

cleanup_fixture() {
  local resolved
  resolved="$(realpath -- "$FIXTURE_ROOT")"
  [[ "$resolved" == /tmp/vit-minpc0-methods-parallel-test.* ]] || return 1
  [[ -d "$resolved" && ! -L "$resolved" ]] || return 1
  rm -rf -- "$resolved"
}
trap cleanup_fixture EXIT

FAKE_PYTHON="$FIXTURE_ROOT/fake-python"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'if [[ "${1:-}" == -c ]]; then exit 0; fi' \
  'state="${FAKE_TRAINING_STATE:?}"' \
  'run_id=' \
  'while [[ $# -gt 0 ]]; do' \
  '  if [[ "$1" == --exp_name ]]; then run_id="$2"; break; fi' \
  '  shift' \
  'done' \
  '[[ -n "$run_id" ]]' \
  'exec 9>"$state/lock"' \
  'flock 9' \
  'active="$(<"$state/active")"' \
  'started="$(<"$state/started")"' \
  'active=$((active + 1))' \
  'started=$((started + 1))' \
  'printf "%s\n" "$active" >"$state/active"' \
  'printf "%s\n" "$started" >"$state/started"' \
  'maximum="$(<"$state/maximum")"' \
  'if ((active > maximum)); then printf "%s\n" "$active" >"$state/maximum"; fi' \
  'printf "%s\n" "$run_id" >>"$state/run_ids"' \
  'flock -u 9' \
  'if [[ "$run_id" == "${FAKE_FAIL_RUN_ID:-}" ]]; then sleep 0.005; status=9; else sleep 0.015; status=0; fi' \
  'flock 9' \
  'active="$(<"$state/active")"' \
  'printf "%s\n" "$((active - 1))" >"$state/active"' \
  'flock -u 9' \
  'if ((status != 0)); then exit "$status"; fi' \
  'printf "%s\n" "Round  100 | acc: 0.5000 | ece: 0.1000 | nll: 1.0000"' \
  'printf "%s\n" "Best: Round 100 | acc: 0.5000"' \
  >"$FAKE_PYTHON"
chmod +x "$FAKE_PYTHON"

FAKE_CHECKPOINT="$FIXTURE_ROOT/old_open_clip_model.safetensors"
FAKE_BIN="$FIXTURE_ROOT/bin"
touch "$FAKE_CHECKPOINT"
mkdir -p "$FAKE_BIN"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'printf "%s  %s\\n" "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f" "${1:-checkpoint}"' \
  >"$FAKE_BIN/sha256sum"
chmod +x "$FAKE_BIN/sha256sum"

initialize_state() {
  local state="$1"
  mkdir -p "$state"
  printf '0\n' >"$state/active"
  printf '0\n' >"$state/started"
  printf '0\n' >"$state/maximum"
  : >"$state/run_ids"
}

run_all() {
  local results_root="$1" legacy_root="$2" state="$3"
  FEDPROREF_VIT_MINPC0_RESULTS_ROOT="$results_root" \
  FEDPROREF_LEGACY_RESULTS_ROOT="$legacy_root" \
  FEDPROREF_DATA_ROOT="$FIXTURE_ROOT/data" \
  FEDPROREF_VIT_PRETRAINED="$FAKE_CHECKPOINT" \
  FEDPROREF_PYTHON_BIN="$FAKE_PYTHON" \
  PATH="$FAKE_BIN:$PATH" \
  FAKE_TRAINING_STATE="$state" \
    bash "$RUNNER" all
}

# Regression: a single dispatcher must run exactly three unique experiments at once.
RESULTS_ROOT="$FIXTURE_ROOT/results"
LEGACY_ROOT="$FIXTURE_ROOT/legacy"
STATE="$FIXTURE_ROOT/state"
initialize_state "$STATE"
run_all "$RESULTS_ROOT" "$LEGACY_ROOT" "$STATE" >"$FIXTURE_ROOT/first-run.log"

observed_maximum="$(<"$STATE/maximum")"
if [[ "$observed_maximum" != 3 ]]; then
  echo "Expected maximum concurrency 3, observed $observed_maximum" >&2
  exit 1
fi
[[ "$(<"$STATE/started")" == 171 ]]
[[ "$(sort -u "$STATE/run_ids" | wc -l)" -eq 171 ]]
[[ "$(find "$RESULTS_ROOT" -mindepth 2 -maxdepth 2 -name .done -type f | wc -l)" -eq 171 ]]
grep -Fxq 'missing=0' "$FIXTURE_ROOT/first-run.log"

# Regression: a resumed all action must skip every valid completed experiment.
run_all "$RESULTS_ROOT" "$LEGACY_ROOT" "$STATE" >"$FIXTURE_ROOT/resume-run.log"
[[ "$(<"$STATE/started")" == 171 ]]
[[ "$(grep -c '^\[SKIP:current\]' "$FIXTURE_ROOT/resume-run.log")" -eq 171 ]]

# Regression: a failed worker must stop the dispatcher from starting new work.
FAIL_RESULTS_ROOT="$FIXTURE_ROOT/fail-results"
FAIL_LEGACY_ROOT="$FIXTURE_ROOT/fail-legacy"
FAIL_STATE="$FIXTURE_ROOT/fail-state"
FAIL_RUN_ID="vit_minpc0_cifar10_a001_c10_sc10_ps42_ts42_fedproref"
initialize_state "$FAIL_STATE"
if FAKE_FAIL_RUN_ID="$FAIL_RUN_ID" \
    run_all "$FAIL_RESULTS_ROOT" "$FAIL_LEGACY_ROOT" "$FAIL_STATE" \
      >"$FIXTURE_ROOT/fail-run.log" 2>&1; then
  echo "Expected the parallel all action to fail when one worker exits nonzero" >&2
  exit 1
fi
[[ "$(<"$FAIL_STATE/started")" == 3 ]]
grep -Fq "[FAIL] $FAIL_RUN_ID (exit=9)" "$FIXTURE_ROOT/fail-run.log"

echo "ViT minpc0 three-worker scheduling: PASS"
