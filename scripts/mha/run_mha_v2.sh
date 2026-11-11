#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m tests.mha.test_mha_paged_accuracy --impl v2 --suite extended \
  --output mha_paged_accuracy_v2_report.json 2>&1 | tee mha_paged_accuracy_v2.log
python -u -m benchmarks.mha.bench_mha_mfu --impl v1 \
  --output mha_mfu_v1_recheck_report.json 2>&1 | tee mha_mfu_v1_recheck.log
python -u -m benchmarks.mha.bench_mha_mfu --impl v2 \
  --baseline-report mha_mfu_v1_recheck_report.json \
  --output mha_mfu_v2_report.json 2>&1 | tee mha_mfu_v2.log
