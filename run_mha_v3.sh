#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u test_mha_paged_accuracy.py --impl v3 --suite cube --patterns full --dtypes bfloat16 --causal noncausal --seeds 7 \
  --output mha_cube_v3_smoke.json 2>&1 | tee mha_cube_v3_smoke.log
python -u test_mha_paged_accuracy.py --impl v3 --suite cube --patterns full --dtypes bfloat16 --causal noncausal \
  --output mha_paged_accuracy_v3_report.json 2>&1 | tee mha_paged_accuracy_v3.log
python -u test_mha_paged_accuracy.py --impl v3 --suite cube --cases cube_decode512 cube_decode2048 \
  --patterns full --dtypes bfloat16 --causal causal --output mha_cube_v3_causal.json 2>&1 | tee mha_cube_v3_causal.log
python -u bench_mha_mfu.py --impl v2 \
  --output mha_mfu_v2_recheck_report.json 2>&1 | tee mha_mfu_v2_recheck.log
python -u bench_mha_mfu.py --impl v3 --baseline-report mha_mfu_v2_recheck_report.json \
  --output mha_mfu_v3_report.json 2>&1 | tee mha_mfu_v3.log
