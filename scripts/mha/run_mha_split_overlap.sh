#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_split_overlap.XXXXXXXX)"
publish() {
  python -u -m benchmarks.mha.summarize_split_overlap "$run_dir"
  printf 'run_directory=%s\n' "$run_dir"
  cat mha_split_overlap_paste.txt
}
trap publish EXIT
python -m unittest tests.mha.test_split_overlap tests.mha.test_prefetch_confirmation -v 2>&1 | tee "$run_dir/host_tests.log"
for kv in 4096 8192; do
  printf '[FGW overlap] timing Q32768 KV%s\n' "$kv"
  python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_kv${kv}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --shape 32768 "$kv" --plan-policy stream --merge-policy rawsimd --raw-baseline-only --only-static-ascendc --split-overlap --rounds 1 --timing-rounds 6 --pairs 8 --initial-warmup 100 --warmup 10 --output "$run_dir/kv${kv}.json" 2>&1 | tee "$run_dir/kv${kv}.log"
  python -u -m benchmarks.mha.summarize_split_overlap "$run_dir"
done
for kv in 4096 8192; do
  printf '[FGW overlap] separate trace Q32768 KV%s\n' "$kv"
  FGW_MAX_TREE_RSS_MB="${FGW_MAX_TREE_RSS_MB:-8192}" python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_trace_kv${kv}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --shape 32768 "$kv" --plan-policy stream --merge-policy rawsimd --raw-baseline-only --only-static-ascendc --split-overlap --split-overlap-trace --rounds 1 --output "$run_dir/trace_kv${kv}.json" 2>&1 | tee "$run_dir/trace_kv${kv}.log"
  python -u -m benchmarks.mha.summarize_split_overlap "$run_dir"
done
