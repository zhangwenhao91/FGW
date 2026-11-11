#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u profile_mha_msprof.py --impl v6 --cases query4096 query32768 \
  --pipeline-stages 2 --buffer-versions 2 --reference-chunk-size 128 \
  --rounds 3 --repeat 30 --warmup 10 --output mha_large_q_msprof_report.json \
  2>&1 | tee mha_large_q_msprof.log
