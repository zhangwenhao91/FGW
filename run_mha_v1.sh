#!/usr/bin/env bash
# Run in the same activated conda/CANN environment as the successful v0 tests.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u test_mha_paged_accuracy.py --impl v1 --suite extended \
  --output mha_paged_accuracy_v1_report.json 2>&1 | tee mha_paged_accuracy_v1.log
# Re-measure v0 on the same device/session before comparing v1.
python -u bench_mha_mfu.py --impl v0 \
  --output mha_mfu_v0_recheck_report.json 2>&1 | tee mha_mfu_v0_recheck.log
python -u bench_mha_mfu.py --impl v1 \
  --baseline-report mha_mfu_v0_recheck_report.json \
  --output mha_mfu_v1_report.json 2>&1 | tee mha_mfu_v1.log
