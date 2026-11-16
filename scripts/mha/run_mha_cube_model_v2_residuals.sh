#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.bench_cube_split_scan --suite holdout-v2 \
  --cases v2_b3h3_q3_d192 v2_b5h1_q16_d64 v2_b3h2_q31_d96 v2_causal_mixed_empty \
  --evaluate-cost-model --general-model-version 2 --compare-model-v1 \
  --splits 1 2 4 8 16 32 --rounds 12 --pairs 61 --warmup 30 \
  --components all --paired-components \
  --component-rounds 12 --component-pairs 61 --component-warmup 30 \
  --save-samples --output mha_cube_model_v2_residuals_summary.json 2>&1 | tee mha_cube_model_v2_residuals.log
