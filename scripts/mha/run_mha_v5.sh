#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m tests.mha.test_mha_paged_accuracy --impl v5 --suite cube --patterns full --dtypes bfloat16 --causal noncausal \
  --output mha_cube_v5_accuracy.json 2>&1 | tee mha_cube_v5_accuracy.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v5 --suite cube --cases cube_decode512 cube_decode2048 \
  --patterns full --dtypes bfloat16 --causal causal --output mha_cube_v5_causal.json 2>&1 | tee mha_cube_v5_causal.log
python -u -m benchmarks.mha.profile_mha_msprof --impl v4 --rounds 2 --output mha_merge_v4_msprof.json 2>&1 | tee mha_merge_v4_msprof.log
python -u -m benchmarks.mha.profile_mha_msprof --impl v5 --rounds 2 --output mha_merge_v5_msprof.json 2>&1 | tee mha_merge_v5_msprof.log
python -u -m benchmarks.mha.bench_mha_mfu --impl v4 --output mha_mfu_v4_merge_baseline.json 2>&1 | tee mha_mfu_v4_merge_baseline.log
python -u -m benchmarks.mha.bench_mha_mfu --impl v5 --baseline-report mha_mfu_v4_merge_baseline.json \
  --output mha_mfu_v5_report.json 2>&1 | tee mha_mfu_v5.log
