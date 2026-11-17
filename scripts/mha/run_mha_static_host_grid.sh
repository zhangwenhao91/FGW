#!/usr/bin/env bash
set -euo pipefail
printf '[FGW] initializing 80-shape static-host grid\n'
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_static_host_grid.XXXXXXXX)"
publish() {
  python -u -m benchmarks.mha.summarize_static_grid "$run_dir"
  printf 'run_directory=%s\n' "$run_dir"
  cat mha_static_host_grid_paste.txt
}
trap publish EXIT
python -m unittest tests.mha.test_static_schedule -v 2>&1 | tee "$run_dir/host_tests.log"
python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_smoke.jsonl" -- python -u -m benchmarks.mha.bench_static_host --smoke --plan-policy cost --output "$run_dir/smoke.json" 2>&1 | tee "$run_dir/smoke.log"
completed=0
started=$SECONDS
for kv in 512 1024 2048 4096 8192 16384 24576 32768; do
  for q in 64 128 256 512 1024 2048 4096 8192 16384 32768; do
    key="q${q}_kv${kv}"
    printf '[FGW grid] starting %d/80 %s elapsed=%ss\n' "$((completed + 1))" "$key" "$((SECONDS - started))"
    python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_${key}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --plan-policy cost --shape "$q" "$kv" --rounds 1 --pairs 3 --warmup 5 --output "$run_dir/$key.json" 2>&1 | tee "$run_dir/$key.log"
    completed=$((completed + 1))
    printf '[FGW grid] completed %d/80 %s elapsed=%ss\n' "$completed" "$key" "$((SECONDS - started))"
    python -u -m benchmarks.mha.summarize_static_grid "$run_dir"
  done
done
