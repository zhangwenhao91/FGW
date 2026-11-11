#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u bench_mha_mfu.py --impl v6 --suite long-q --dtypes bfloat16 --causal noncausal \
  --warmup 20 --repeat 51 --output mha_mfu_v6_long_q_report.json 2>&1 | tee mha_mfu_v6_long_q.log
python -u bench_mha_mfu.py --impl v6 --suite long-q --dtypes bfloat16 --causal noncausal --reverse \
  --warmup 20 --repeat 51 --output mha_mfu_v6_long_q_reverse_report.json 2>&1 | tee mha_mfu_v6_long_q_reverse.log
