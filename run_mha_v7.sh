#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u test_mha_paged_accuracy.py --impl v7 --suite cube --cases cube_direct_q1 cube_direct_q129 \
  --patterns full --dtypes bfloat16 --causal noncausal --seeds 7 \
  --output mha_v7_direct_smoke.json 2>&1 | tee mha_v7_direct_smoke.log
python -u test_mha_paged_accuracy.py --impl v7 --suite cube --patterns full --dtypes bfloat16 --causal noncausal \
  --output mha_cube_v7_accuracy.json 2>&1 | tee mha_cube_v7_accuracy.log
python -u test_mha_paged_accuracy.py --impl v7 --suite cube --cases cube_decode512 cube_decode2048 cube_direct_q1 \
  --patterns full --dtypes bfloat16 --causal causal --output mha_cube_v7_causal.json 2>&1 | tee mha_cube_v7_causal.log
python -u profile_mha_msprof.py --impl v7 --cases query4096 query32768 \
  --rounds 3 --repeat 30 --warmup 10 --output mha_v7_large_q_msprof.json 2>&1 | tee mha_v7_large_q_msprof.log
python -u bench_mha_mfu.py --impl v7 --suite large-q --dtypes bfloat16 --causal noncausal \
  --reference-chunk-size 128 --warmup 20 --repeat 51 \
  --output mha_mfu_v7_large_q_report.json 2>&1 | tee mha_mfu_v7_large_q.log
python -u bench_mha_mfu.py --impl v7 --suite large-q --dtypes bfloat16 --causal noncausal --reverse \
  --reference-chunk-size 128 --warmup 20 --repeat 51 \
  --output mha_mfu_v7_large_q_reverse_report.json 2>&1 | tee mha_mfu_v7_large_q_reverse.log
