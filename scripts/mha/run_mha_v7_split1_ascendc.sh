#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m benchmarks.mha.bench_v7_ascendc --split-comparison --fia-api v2 \
  --kv 8192 --q-lengths 1 129 4096 --rounds 1 --warmup 5 --repeat 11 \
  --output mha_v7_split1_ascendc_smoke.json 2>&1 | tee mha_v7_split1_ascendc_smoke.log
failed=0
for kv in 2048 512 8192 32768; do
  query_args=()
  if [ "$kv" != 2048 ]; then query_args=(--q-lengths 1 16 128 129 512 1024 4096 8192); fi
  name="mha_v7_split1_ascendc_kv${kv}"
  if ! python -u -m benchmarks.mha.bench_v7_ascendc --split-comparison --fia-api v2 \
    --kv "$kv" "${query_args[@]}" --rounds 4 --warmup 20 --repeat 51 \
    --output "${name}.json" 2>&1 | tee "${name}.log"; then
    failed=1
  fi
done
exit "$failed"
