#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
failed=0
for sweep in 1 2; do
  if [ "$sweep" = 1 ]; then candidates="base resident prefetch combined"; else candidates="combined prefetch resident base"; fi
  for candidate in $candidates; do
    flags=(--no-q-resident)
    case "$candidate" in
      resident) flags=(--q-resident) ;;
      prefetch) flags=(--no-q-resident --pipeline-stages 3 --kv-versions 3) ;;
      combined) flags=(--q-resident --pipeline-stages 3 --kv-versions 3) ;;
    esac
    name="mha_overlap_${candidate}_run${sweep}"
    if ! python -u test_mha_paged_accuracy.py --impl v7 --suite cube --patterns full --dtypes bfloat16 \
      --causal noncausal "${flags[@]}" --output "${name}_accuracy.json" 2>&1 | tee "${name}_accuracy.log"; then
      failed=1; continue
    fi
    if ! python -u test_mha_paged_accuracy.py --impl v7 --suite cube --cases cube_decode512 cube_direct_q1 \
      --patterns full --dtypes bfloat16 --causal causal "${flags[@]}" \
      --output "${name}_causal.json" 2>&1 | tee "${name}_causal.log"; then
      failed=1; continue
    fi
    if ! python -u profile_mha_msprof.py --impl v7 --cases query4096 query32768 "${flags[@]}" \
      --rounds 2 --repeat 30 --warmup 10 --output "${name}_msprof.json" 2>&1 | tee "${name}_msprof.log"; then
      failed=1; continue
    fi
    if ! python -u bench_mha_mfu.py --impl v7 --suite large-q --cases q4096_b2_h8_d128_k2048 q32768_b2_h8_d128_k2048 \
      --dtypes bfloat16 --causal noncausal --reference-chunk-size 128 --warmup 20 --repeat 51 "${flags[@]}" \
      --output "${name}_mfu.json" 2>&1 | tee "${name}_mfu.log"; then
      failed=1
    fi
  done
done
exit "$failed"
