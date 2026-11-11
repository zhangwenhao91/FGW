#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m tests.mha.test_mha_paged_accuracy --impl v3 --suite cube --patterns full --dtypes bfloat16 --causal noncausal --seeds 7 \
  --output mha_cube_v3_smoke.json 2>&1 | tee mha_cube_v3_smoke.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v3 --suite cube --patterns full --dtypes bfloat16 --causal noncausal \
  --output mha_paged_accuracy_v3_report.json 2>&1 | tee mha_paged_accuracy_v3.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v3 --suite cube --cases cube_decode512 cube_decode2048 \
  --patterns full --dtypes bfloat16 --causal causal --output mha_cube_v3_causal.json 2>&1 | tee mha_cube_v3_causal.log
python -u -m benchmarks.mha.bench_mha_mfu --impl v2 \
  --output mha_mfu_v2_recheck_report.json 2>&1 | tee mha_mfu_v2_recheck.log
python -u -m benchmarks.mha.bench_mha_mfu --impl v3 --baseline-report mha_mfu_v2_recheck_report.json \
  --output mha_mfu_v3_report.json 2>&1 | tee mha_mfu_v3.log
