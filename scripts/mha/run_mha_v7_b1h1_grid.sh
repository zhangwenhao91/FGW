#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
# A fresh directory prevents partial reruns from mixing with older KV reports.
run_dir="$(mktemp -d mha_v7_b1h1_grid.XXXXXXXX)"
trap 'python -m benchmarks.mha.summarize_v7_b1h1_grid "$run_dir"' EXIT
python -u -m benchmarks.mha.bench_v7_ascendc --fia-api v2 --bh 1x1 \
  --kv 512 --q-lengths 1 129 --rounds 2 --warmup 5 --repeat 11 \
  --output "$run_dir/smoke.json" 2>&1 | tee "$run_dir/smoke.log"
for kv in 512 2048 4096 8192; do
  python -u -m benchmarks.mha.bench_v7_ascendc --fia-api v2 --bh 1x1 \
    --kv "$kv" --q-lengths 1 4 16 32 64 127 128 129 256 512 1024 2048 4096 8192 \
    --rounds 6 --warmup 30 --repeat 51 \
    --output "$run_dir/kv${kv}.json" 2>&1 | tee "$run_dir/kv${kv}.log"
done
