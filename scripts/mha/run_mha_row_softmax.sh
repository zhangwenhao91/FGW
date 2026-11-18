#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_row_softmax.XXXXXXXX)"
publish() {
  python -u -m benchmarks.mha.summarize_row_softmax "$run_dir"
  printf 'run_directory=%s\n' "$run_dir"
  cat mha_row_softmax_paste.txt
}
trap publish EXIT
python -m unittest tests.mha.test_static_schedule tests.mha.test_prefetch_confirmation tests.mha.test_row_softmax_report -v 2>&1 | tee "$run_dir/host_tests.log"
for kv in 4096 8192; do
  printf '[FGW row softmax register_rows2_v2] Q32768 KV%s\n' "$kv"
  python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_kv${kv}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --shape 32768 "$kv" --plan-policy stream --merge-policy rawsimd --row-softmax --only-static-ascendc --rounds 1 --timing-rounds 6 --pairs 9 --initial-warmup 100 --warmup 10 --output "$run_dir/kv${kv}.json" 2>&1 | tee "$run_dir/kv${kv}.log"
  python -u -m benchmarks.mha.summarize_row_softmax "$run_dir"
done
