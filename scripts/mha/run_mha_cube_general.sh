#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
# Fail fast on compilation/runtime failures before starting the full sweep.
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --suite general --seeds 7 \
  --cases general_d16_page16 general_d48_page48 general_d128_q129 general_d256 \
  --output mha_cube_general_smoke.json 2>&1 | tee mha_cube_general_smoke.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --suite general --output mha_cube_general_boundaries.json 2>&1 | tee mha_cube_general_boundaries.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --suite extended --output mha_cube_general_accuracy.json 2>&1 | tee mha_cube_general_accuracy.log
