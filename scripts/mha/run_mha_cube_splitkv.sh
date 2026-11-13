#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --suite general --seeds 7 \
  --cases general_d16_page16 general_d48_page48 general_d128_q129 general_d256 \
  --output mha_cube_splitkv_smoke.json 2>&1 | tee mha_cube_splitkv_smoke.log
for splits in 1 2 4 8 16 32; do
  python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
    --suite split --seeds 7 42 --num-splits "$splits" \
    --output "mha_cube_splitkv_s${splits}_accuracy.json" 2>&1 | tee "mha_cube_splitkv_s${splits}_accuracy.log"
done
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --suite general --output mha_cube_splitkv_boundaries.json 2>&1 | tee mha_cube_splitkv_boundaries.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --cube-mode general \
  --suite extended --output mha_cube_splitkv_accuracy.json 2>&1 | tee mha_cube_splitkv_accuracy.log
python -u -m benchmarks.mha.bench_cube_ascendc --suite all \
  --cube-splits auto --compare-split1 --rounds 6 --pairs 31 --warmup 20 \
  --output mha_cube_splitkv_mfu_summary.json 2>&1 | tee mha_cube_splitkv_mfu.log
