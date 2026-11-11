#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u bench_mha_mfu.py --impl v6 --suite large-q --dtypes bfloat16 --causal noncausal \
  --reference-chunk-size 128 --warmup 20 --repeat 51 \
  --output mha_mfu_v6_large_q_report.json 2>&1 | tee mha_mfu_v6_large_q.log
python -u bench_mha_mfu.py --impl v6 --suite large-q --dtypes bfloat16 --causal noncausal --reverse \
  --reference-chunk-size 128 --warmup 20 --repeat 51 \
  --output mha_mfu_v6_large_q_reverse_report.json 2>&1 | tee mha_mfu_v6_large_q_reverse.log
