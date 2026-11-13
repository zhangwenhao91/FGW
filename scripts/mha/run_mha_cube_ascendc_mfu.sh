#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.bench_cube_ascendc --suite standard \
  --cases dim64_q16 causal_d128_q129 --rounds 2 --pairs 7 --warmup 5 \
  --output mha_cube_ascendc_mfu_smoke_summary.json 2>&1 | tee mha_cube_ascendc_mfu_smoke.log
python -u -m benchmarks.mha.bench_cube_ascendc --suite all \
  --rounds 6 --pairs 31 --warmup 20 \
  --output mha_cube_ascendc_mfu_summary.json 2>&1 | tee mha_cube_ascendc_mfu.log
