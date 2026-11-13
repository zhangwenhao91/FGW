#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
model_run_started="$(python -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())')"
export_status() {
  python -m tools.mha.summarize_general_model --since "$model_run_started"
}
trap export_status EXIT
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --general-split-policy cost_model --suite general \
  --output mha_cube_model_boundaries.json 2>&1 | tee mha_cube_model_boundaries.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --general-split-policy cost_model --suite extended \
  --output mha_cube_model_accuracy.json 2>&1 | tee mha_cube_model_accuracy.log
python -u -m benchmarks.mha.bench_cube_split_scan --suite standard --evaluate-cost-model \
  --components none --rounds 6 --pairs 31 --warmup 20 \
  --output mha_cube_model_replay_summary.json 2>&1 | tee mha_cube_model_replay.log
python -u -m benchmarks.mha.bench_cube_split_scan --suite holdout --evaluate-cost-model \
  --components none --rounds 6 --pairs 31 --warmup 20 \
  --output mha_cube_model_holdout_summary.json 2>&1 | tee mha_cube_model_holdout.log
