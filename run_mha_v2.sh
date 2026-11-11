#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u test_mha_paged_accuracy.py --impl v2 --suite extended \
  --output mha_paged_accuracy_v2_report.json 2>&1 | tee mha_paged_accuracy_v2.log
python -u bench_mha_mfu.py --impl v1 \
  --output mha_mfu_v1_recheck_report.json 2>&1 | tee mha_mfu_v1_recheck.log
python -u bench_mha_mfu.py --impl v2 \
  --baseline-report mha_mfu_v1_recheck_report.json \
  --output mha_mfu_v2_report.json 2>&1 | tee mha_mfu_v2.log
