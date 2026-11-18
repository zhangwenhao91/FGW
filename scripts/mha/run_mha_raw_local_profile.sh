#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
# Export spawns Python/msprof children; summed RSS exceeded 4 GiB on the NPU
# host even though all five local calls had finished. Scope this larger budget
# to profiling, retaining the guard and any explicit operator override.
export FGW_MAX_TREE_RSS_MB="${FGW_MAX_TREE_RSS_MB:-8192}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_raw_local_profile.XXXXXXXX)"
publish() {
  printf 'run_directory=%s\n' "$run_dir" > mha_raw_local_profile_paste.txt
  for kv in 4096 8192; do
    if [ -f "$run_dir/kv${kv}_paste.txt" ]; then
      cat "$run_dir/kv${kv}_paste.txt" >> mha_raw_local_profile_paste.txt
    fi
    if [ -f "$run_dir/guard_kv${kv}.jsonl" ]; then
      printf 'KV%s guard_last_record: ' "$kv" >> mha_raw_local_profile_paste.txt
      tail -n 1 "$run_dir/guard_kv${kv}.jsonl" >> mha_raw_local_profile_paste.txt
    fi
  done
  cat mha_raw_local_profile_paste.txt
}
trap publish EXIT
python -m unittest tests.mha.test_local_profile_parser -v 2>&1 | tee "$run_dir/parser_tests.log"
for kv in 4096 8192; do
  printf '[FGW profiling] Q32768 KV%s\n' "$kv"
  python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_kv${kv}.jsonl" -- python -u -m benchmarks.mha.profile_raw_local --kv "$kv" --sessions 3 --repeat 5 --warmup 100 --output "$run_dir/kv${kv}.json" 2>&1 | tee "$run_dir/kv${kv}.log"
done
