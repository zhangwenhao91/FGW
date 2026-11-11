#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u test_mha_paged_accuracy.py --impl v4 --suite cube --cases cube_query4 cube_q129 \
  --patterns full --dtypes bfloat16 --causal noncausal --seeds 7 \
  --output mha_cube_v4_smoke.json 2>&1 | tee mha_cube_v4_smoke.log
python -u test_mha_paged_accuracy.py --impl v4 --suite cube --patterns full --dtypes bfloat16 --causal noncausal \
  --output mha_cube_v4_accuracy.json 2>&1 | tee mha_cube_v4_accuracy.log
python -u test_mha_paged_accuracy.py --impl v4 --suite cube --cases cube_decode512 cube_decode2048 \
  --patterns full --dtypes bfloat16 --causal causal --output mha_cube_v4_causal.json 2>&1 | tee mha_cube_v4_causal.log
python -u bench_mha_mfu.py --impl v3 --output mha_mfu_v3_recheck_report.json 2>&1 | tee mha_mfu_v3_recheck.log
python -u bench_mha_mfu.py --impl v4 --baseline-report mha_mfu_v3_recheck_report.json \
  --output mha_mfu_v4_report.json 2>&1 | tee mha_mfu_v4.log
