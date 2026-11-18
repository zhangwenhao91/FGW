#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_standalone_fused_grid.XXXXXXXX)"
publish() {
  python -u -m benchmarks.mha.summarize_raw_grid "$run_dir" --standalone-fused
  printf 'run_directory=%s\n' "$run_dir"
  cat mha_standalone_fused_grid_paste.txt
}
trap publish EXIT
python -m unittest tests.mha.test_static_schedule tests.mha.test_raw_partial_state tests.mha.test_raw_grid_report -v 2>&1 | tee "$run_dir/host_tests.log"
completed=0
started=$SECONDS
for kv in 512 1024 2048 4096 8192 16384 24576 32768; do
  for q in 64 128 256 512 1024 2048 4096 8192 16384 32768; do
    key="q${q}_kv${kv}"
    printf '[FGW fused grid] starting %d/80 %s elapsed=%ss\n' "$((completed + 1))" "$key" "$((SECONDS - started))"
    python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_${key}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --shape "$q" "$kv" --plan-policy stream --merge-policy rawsimd --raw-baseline-only --standalone-fused --only-static-ascendc --rounds 1 --timing-rounds 6 --pairs 4 --initial-warmup 100 --warmup 10 --output "$run_dir/$key.json" 2>&1 | tee "$run_dir/$key.log"
    completed=$((completed + 1))
    printf '[FGW fused grid] completed %d/80 %s elapsed=%ss\n' "$completed" "$key" "$((SECONDS - started))"
    python -u -m benchmarks.mha.summarize_raw_grid "$run_dir" --standalone-fused
  done
done
