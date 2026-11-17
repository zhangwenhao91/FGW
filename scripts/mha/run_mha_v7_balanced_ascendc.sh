#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_v7_balanced_ascendc.XXXXXXXX)"
publish() {
  if [ -f "$run_dir/full.json" ]; then
    python -m benchmarks.mha.summarize_clean_grid "$run_dir/full.json" "$run_dir/grid_paste.txt" || true
  fi
  printf 'run_directory=%s\n' "$run_dir" > mha_v7_balanced_ascendc_paste.txt
  if [ -f "$run_dir/grid_paste.txt" ]; then
    cat "$run_dir/grid_paste.txt" >> mha_v7_balanced_ascendc_paste.txt
  elif [ -f "$run_dir/full_paste.txt" ]; then
    cat "$run_dir/full_paste.txt" >> mha_v7_balanced_ascendc_paste.txt
  fi
  cat mha_v7_balanced_ascendc_paste.txt
}
trap publish EXIT
python -u -m benchmarks.mha.bench_kv_prefetch --balanced-grid --pairs 7 --warmup 5 \
  --output "$run_dir/full.json" 2>&1 | tee "$run_dir/full.log"
