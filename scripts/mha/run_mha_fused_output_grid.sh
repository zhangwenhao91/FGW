#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
if [ "$#" -eq 0 ]; then
  run_dir="$(mktemp -d "$repo_root/reports/mha/latest/mha_fused_output_grid.XXXXXXXX")"
elif [ "$#" -eq 2 ] && [ "$1" = --resume ]; then
  run_dir="$(realpath -e -- "$2")"
  [ -d "$run_dir" ] || exit 2
else
  printf 'Usage: bash scripts/mha/run_mha_fused_output_grid.sh [--resume RUN_DIRECTORY]\n' >&2
  exit 2
fi
publish() {
  python -u -m benchmarks.mha.summarize_fused_output_grid "$run_dir"
  printf 'run_directory=%s\n' "$run_dir"
  cat "$repo_root/reports/mha/latest/mha_fused_output_grid_paste.txt"
}
trap publish EXIT
printf 'run_directory=%s\n' "$run_dir"
cd -- "$repo_root"
python -m unittest tests.mha.test_fused_output_report tests.mha.test_fused_output_grid tests.mha.test_prefetch_confirmation -v 2>&1 | tee "$run_dir/host_tests.log"
completed=0
started=$SECONDS
for kv in 512 1024 2048 4096 8192 16384 24576 32768; do
  for q in 64 128 256 512 1024 2048 4096 8192 16384 32768; do
    key="q${q}_kv${kv}"
    if python -m benchmarks.mha.summarize_fused_output_grid "$run_dir" --check-case "$q" "$kv"; then
      completed=$((completed+1))
      printf '[FGW fused grid] reused %d/80 %s\n' "$completed" "$key"
      continue
    fi
    printf '[FGW fused grid] starting %d/80 %s elapsed=%ss\n' "$((completed+1))" "$key" "$((SECONDS-started))"
    python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_${key}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --shape "$q" "$kv" --plan-policy stream --merge-policy rawsimd --fused-output --only-static-ascendc --rounds 1 --timing-rounds 6 --pairs 9 --initial-warmup 100 --warmup 10 --output "$run_dir/$key.json" 2>&1 | tee "$run_dir/$key.log"
    completed=$((completed+1))
    printf '[FGW fused grid] completed %d/80 %s elapsed=%ss\n' "$completed" "$key" "$((SECONDS-started))"
    python -u -m benchmarks.mha.summarize_fused_output_grid "$run_dir"
  done
done
