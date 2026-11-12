#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.bench_v7_split_scan --evaluate-cost-model --kv 4096 \
  --bh 1x1 1x8 --q-lengths 1 129 --rounds 1 --warmup 5 --repeat 11 \
  --output mha_split_model_smoke.json 2>&1 | tee mha_split_model_smoke.log
failed=0
for kv in 4096 16384; do
  name="mha_split_model_holdout_kv${kv}"
  if ! python -u -m benchmarks.mha.bench_v7_split_scan --evaluate-cost-model --kv "$kv" \
    --bh 1x1 1x8 4x4 2x16 --q-lengths 1 64 129 256 512 2048 8192 \
    --rounds 4 --warmup 20 --repeat 51 --output "${name}.json" 2>&1 | tee "${name}.log"; then
    failed=1
  fi
done
for kv in 2048 32768; do
  name="mha_split_model_boundary_kv${kv}"
  if ! python -u -m benchmarks.mha.bench_v7_split_scan --evaluate-cost-model --kv "$kv" \
    --bh 1x1 2x8 --q-lengths 1 128 129 1024 \
    --rounds 4 --warmup 20 --repeat 51 --output "${name}.json" 2>&1 | tee "${name}.log"; then
    failed=1
  fi
done
exit "$failed"
