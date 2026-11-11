#!/usr/bin/env bash
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
failed=0
for sweep in 1 2; do
  if [ "$sweep" = 1 ]; then modes="auto auto-offset manual"; else modes="manual auto-offset auto"; fi
  for mode in $modes; do
    name="mha_schedule_${mode}_run${sweep}"
    if ! python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --suite cube \
      --patterns full --dtypes bfloat16 --causal noncausal --schedule-mode "$mode" \
      --output "${name}_accuracy.json" 2>&1 | tee "${name}_accuracy.log"; then
      failed=1; continue
    fi
    if ! python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --suite cube \
      --cases cube_decode512 cube_decode2048 cube_direct_q1 --patterns full --dtypes bfloat16 \
      --causal causal --schedule-mode "$mode" --output "${name}_causal.json" 2>&1 | tee "${name}_causal.log"; then
      failed=1; continue
    fi
    if ! python -u -m benchmarks.mha.profile_mha_msprof --impl v7 --cases query4096 query32768 \
      --schedule-mode "$mode" --rounds 2 --repeat 30 --warmup 10 \
      --output "${name}_msprof.json" 2>&1 | tee "${name}_msprof.log"; then
      failed=1; continue
    fi
    if ! python -u -m benchmarks.mha.bench_mha_mfu --impl v7 --suite large-q \
      --cases q4096_b2_h8_d128_k2048 q32768_b2_h8_d128_k2048 --dtypes bfloat16 --causal noncausal \
      --reference-chunk-size 128 --warmup 20 --repeat 51 --schedule-mode "$mode" \
      --output "${name}_mfu.json" 2>&1 | tee "${name}_mfu.log"; then
      failed=1
    fi
  done
done
exit "$failed"
