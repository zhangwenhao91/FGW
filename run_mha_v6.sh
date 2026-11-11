#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u test_mha_paged_accuracy.py --impl v6 --suite cube --patterns full --dtypes bfloat16 --causal noncausal \
  --output mha_cube_v6_accuracy.json 2>&1 | tee mha_cube_v6_accuracy.log
python -u test_mha_paged_accuracy.py --impl v6 --suite cube --cases cube_decode512 cube_decode2048 \
  --patterns full --dtypes bfloat16 --causal causal --output mha_cube_v6_causal.json 2>&1 | tee mha_cube_v6_causal.log
python -u profile_mha_msprof.py --impl v5 --rounds 2 --output mha_merge_v5_recheck.json 2>&1 | tee mha_merge_v5_recheck.log
python -u profile_mha_msprof.py --impl v6 --rounds 2 --output mha_merge_v6_msprof.json 2>&1 | tee mha_merge_v6_msprof.log
python -u bench_mha_mfu.py --impl v5 --output mha_mfu_v5_recheck.json 2>&1 | tee mha_mfu_v5_recheck.log
python -u bench_mha_mfu.py --impl v6 --baseline-report mha_mfu_v5_recheck.json \
  --output mha_mfu_v6_report.json 2>&1 | tee mha_mfu_v6.log
