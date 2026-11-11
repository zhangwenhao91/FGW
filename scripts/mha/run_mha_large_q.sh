#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.bench_mha_mfu --impl v6 --suite large-q --dtypes bfloat16 --causal noncausal \
  --reference-chunk-size 128 --warmup 20 --repeat 51 \
  --output mha_mfu_v6_large_q_report.json 2>&1 | tee mha_mfu_v6_large_q.log
python -u -m benchmarks.mha.bench_mha_mfu --impl v6 --suite large-q --dtypes bfloat16 --causal noncausal --reverse \
  --reference-chunk-size 128 --warmup 20 --repeat 51 \
  --output mha_mfu_v6_large_q_reverse_report.json 2>&1 | tee mha_mfu_v6_large_q_reverse.log
