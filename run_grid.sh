#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports"
run_dir="$(mktemp -d "$repo_root/reports/grid.XXXXXXXX")"
publish() {
  python -u "$repo_root/benchmark.py" summarize "$run_dir"
  printf 'run_directory=%s\n' "$run_dir"
  cat "$repo_root/reports/mha_50fca6d_grid_paste.txt"
}
trap publish EXIT
cd -- "$repo_root"
python benchmark.py verify
python -m unittest test_benchmark -v
completed=0
started=$SECONDS
for kv in 512 1024 2048 4096 8192 16384 24576 32768; do
  for q in 64 128 256 512 1024 2048 4096 8192 16384 32768; do
    key="q${q}_kv${kv}"
    printf '[50fca6d grid] starting %d/80 %s elapsed=%ss\n' "$((completed+1))" "$key" "$((SECONDS-started))"
    python -u resource_guard.py --log "$run_dir/guard_${key}.jsonl" -- python -u benchmark.py case --q "$q" --kv "$kv" --output "$run_dir/$key.json" 2>&1 | tee "$run_dir/$key.log"
    completed=$((completed+1))
    printf '[50fca6d grid] completed %d/80 %s elapsed=%ss\n' "$completed" "$key" "$((SECONDS-started))"
    python -u benchmark.py summarize "$run_dir"
  done
done
