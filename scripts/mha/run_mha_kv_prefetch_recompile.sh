#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_kv_prefetch_recompile.XXXXXXXX)"
publish() {
  printf 'run_directory=%s\n' "$run_dir" > mha_kv_prefetch_recompile_paste.txt
  for phase in smoke full; do
    if [ -f "$run_dir/${phase}_paste.txt" ]; then
      printf '\nphase=%s\n' "$phase" >> mha_kv_prefetch_recompile_paste.txt
      cat "$run_dir/${phase}_paste.txt" >> mha_kv_prefetch_recompile_paste.txt
    fi
  done
  cat mha_kv_prefetch_recompile_paste.txt
}
trap publish EXIT
python -u -m benchmarks.mha.bench_kv_prefetch --quick-regressions \
  --warmup 5 --output "$run_dir/full.json" 2>&1 | tee "$run_dir/full.log"
