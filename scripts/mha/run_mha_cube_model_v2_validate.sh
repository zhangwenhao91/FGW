#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
model_run_started="$(python -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())')"
export_status() {
  python -m tools.mha.summarize_general_model --since "$model_run_started" --output mha_cube_model_v2_status.txt
}
trap export_status EXIT
# Kernel math is unchanged; check the new host selector through the public API.
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --general-split-policy cost_model --general-model-version 2 --suite general --seeds 7 \
  --output mha_cube_model_v2_accuracy.json 2>&1 | tee mha_cube_model_v2_accuracy.log
python -u -m benchmarks.mha.bench_cube_split_scan --suite expanded --evaluate-cost-model \
  --general-model-version 2 --compare-model-v1 --components none --rounds 6 --pairs 31 --warmup 20 \
  --output mha_cube_model_v2_replay_summary.json 2>&1 | tee mha_cube_model_v2_replay.log
python -u -m benchmarks.mha.bench_cube_split_scan --suite holdout-v2 --evaluate-cost-model \
  --general-model-version 2 --compare-model-v1 --components none --rounds 6 --pairs 31 --warmup 20 \
  --output mha_cube_model_v2_holdout_summary.json 2>&1 | tee mha_cube_model_v2_holdout.log
