#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# This targeted smoke includes actual Cube cases (quick suite does not).
python -u test_mha_paged_accuracy.py --impl v3 --suite extended --cases page256 --seeds 7 \
  --output mha_cube_v3_smoke.json 2>&1 | tee mha_cube_v3_smoke.log
python -u test_mha_paged_accuracy.py --impl v3 --suite extended \
  --output mha_paged_accuracy_v3_report.json 2>&1 | tee mha_paged_accuracy_v3.log
python -u bench_mha_mfu.py --impl v2 \
  --output mha_mfu_v2_recheck_report.json 2>&1 | tee mha_mfu_v2_recheck.log
python -u bench_mha_mfu.py --impl v3 --baseline-report mha_mfu_v2_recheck_report.json \
  --output mha_mfu_v3_report.json 2>&1 | tee mha_mfu_v3.log
