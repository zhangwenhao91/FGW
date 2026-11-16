#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
trap 'python -m benchmarks.mha.summarize_external_gather' EXIT
python -u -m benchmarks.mha.bench_external_gather --suite standard \
  --cases dim64_q16 gather_fast_page16 gather_fast_page64 gather_zero_mixed \
  --cube-splits auto --rounds 2 --pairs 7 --warmup 5 \
  --output mha_external_gather_smoke_summary.json 2>&1 | tee mha_external_gather_smoke.log
python -u -m benchmarks.mha.bench_external_gather --suite all \
  --cube-splits auto --rounds 6 --pairs 31 --warmup 20 \
  --output mha_external_gather_summary.json 2>&1 | tee mha_external_gather.log
