#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u test_mha_paged_accuracy.py --impl v5 --suite cube --patterns full --dtypes bfloat16 --causal noncausal \
  --output mha_cube_v5_accuracy.json 2>&1 | tee mha_cube_v5_accuracy.log
python -u test_mha_paged_accuracy.py --impl v5 --suite cube --cases cube_decode512 cube_decode2048 \
  --patterns full --dtypes bfloat16 --causal causal --output mha_cube_v5_causal.json 2>&1 | tee mha_cube_v5_causal.log
python -u profile_mha_msprof.py --impl v4 --rounds 2 --output mha_merge_v4_msprof.json 2>&1 | tee mha_merge_v4_msprof.log
python -u profile_mha_msprof.py --impl v5 --rounds 2 --output mha_merge_v5_msprof.json 2>&1 | tee mha_merge_v5_msprof.log
python -u bench_mha_mfu.py --impl v4 --output mha_mfu_v4_merge_baseline.json 2>&1 | tee mha_mfu_v4_merge_baseline.log
python -u bench_mha_mfu.py --impl v5 --baseline-report mha_mfu_v4_merge_baseline.json \
  --output mha_mfu_v5_report.json 2>&1 | tee mha_mfu_v5.log
