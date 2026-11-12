#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.bench_v7_ascendc --q-lengths 1 128 4096 --rounds 1 --warmup 5 --repeat 11 \
  --output mha_v7_ascendc_smoke.json 2>&1 | tee mha_v7_ascendc_smoke.log
python -u -m benchmarks.mha.bench_v7_ascendc --rounds 2 --warmup 20 --repeat 51 \
  --output mha_v7_ascendc_speedup.json 2>&1 | tee mha_v7_ascendc_speedup.log
