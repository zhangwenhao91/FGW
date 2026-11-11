#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
for sweep in 1 2; do
  reverse=()
  if [ "$sweep" = 1 ]; then candidates="base resident"; else candidates="resident base"; reverse=(--reverse); fi
  for candidate in $candidates; do
    flags=(--no-q-resident)
    if [ "$candidate" = resident ]; then flags=(--q-resident); fi
    name="mha_short_q_${candidate}_run${sweep}"
    python -u -m benchmarks.mha.bench_mha_mfu --impl v7 --suite short-q --dtypes bfloat16 --causal noncausal \
      --warmup 20 --repeat 51 "${flags[@]}" "${reverse[@]}" \
      --output "${name}_noncausal.json" 2>&1 | tee "${name}_noncausal.log"
    python -u -m benchmarks.mha.bench_mha_mfu --impl v7 --suite short-q --cases q1_b2_h8_d128_k2048 decode_b1_k512 \
      --dtypes bfloat16 --causal causal --warmup 20 --repeat 51 "${flags[@]}" "${reverse[@]}" \
      --output "${name}_causal.json" 2>&1 | tee "${name}_causal.log"
  done
done
