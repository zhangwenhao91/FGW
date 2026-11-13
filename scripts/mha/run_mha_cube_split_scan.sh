#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.bench_cube_split_scan \
  --splits 1 2 4 8 16 32 --rounds 6 --pairs 31 --warmup 20 \
  --components focus --component-rounds 2 --component-pairs 11 --component-warmup 5 \
  --output mha_cube_split_scan_summary.json 2>&1 | tee mha_cube_split_scan.log
