#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.bench_v7_split_scan --kv 2048 --q-lengths 1 129 \
  --rounds 1 --warmup 5 --repeat 11 --output mha_v7_split_scan_smoke.json \
  2>&1 | tee mha_v7_split_scan_smoke.log
failed=0
for kv in 512 2048 8192 32768; do
  name="mha_v7_split_scan_kv${kv}"
  if ! python -u -m benchmarks.mha.bench_v7_split_scan --kv "$kv" \
    --rounds 4 --warmup 20 --repeat 51 --output "${name}.json" \
    2>&1 | tee "${name}.log"; then
    failed=1
  fi
done
exit "$failed"
