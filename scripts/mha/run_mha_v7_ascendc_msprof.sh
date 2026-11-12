#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.profile_v7_ascendc_msprof --rounds 3 --warmup 20 \
  --repeat 30 --event-repeat 51 --cache-mb 256 --output mha_v7_ascendc_msprof.json \
  2>&1 | tee mha_v7_ascendc_msprof.log
