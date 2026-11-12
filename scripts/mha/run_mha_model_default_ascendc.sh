#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --suite cube \
  --patterns full --dtypes bfloat16 --causal noncausal --output mha_model_default_accuracy.json \
  2>&1 | tee mha_model_default_accuracy.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --suite cube \
  --cases cube_decode512 cube_decode2048 cube_direct_q1 --patterns full --dtypes bfloat16 \
  --causal causal --output mha_model_default_causal.json 2>&1 | tee mha_model_default_causal.log
python -u -m benchmarks.mha.bench_v7_ascendc --compare-legacy --fia-api v2 --kv 4096 \
  --bh 1x1 1x8 --q-lengths 1 129 --rounds 1 --warmup 5 --repeat 11 \
  --output mha_model_default_smoke.json 2>&1 | tee mha_model_default_smoke.log
failed=0
for kv in 512 2048 4096 8192 16384 32768; do
  query_args=(--q-lengths 1 16 128 129 512 1024 4096 8192)
  bh_args=(--bh 2x8 1x1)
  if [ "$kv" = 2048 ]; then query_args=(); fi
  if [ "$kv" = 4096 ] || [ "$kv" = 16384 ]; then
    query_args=(--q-lengths 1 64 129 256 512 2048 8192)
    bh_args=(--bh 1x1 1x8 4x4 2x16)
  fi
  name="mha_model_default_kv${kv}"
  if ! python -u -m benchmarks.mha.bench_v7_ascendc --compare-legacy --fia-api v2 \
    --kv "$kv" "${query_args[@]}" "${bh_args[@]}" --rounds 4 --warmup 20 --repeat 51 \
    --output "${name}.json" 2>&1 | tee "${name}.log"; then
    failed=1
  fi
done
exit "$failed"
