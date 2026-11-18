#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_fused_output.XXXXXXXX)"
publish() {
  python -u -m benchmarks.mha.summarize_fused_output "$run_dir"
  printf 'run_directory=%s\n' "$run_dir"
  cat mha_fused_output_paste.txt
}
trap publish EXIT
python -m unittest tests.mha.test_fused_output_report tests.mha.test_prefetch_confirmation -v 2>&1 | tee "$run_dir/host_tests.log"
for kv in 512 1024 2048 4096 8192; do
  printf '[FGW fused output] Q32768 KV%s\n' "$kv"
  python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_kv${kv}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --shape 32768 "$kv" --plan-policy stream --merge-policy rawsimd --fused-output --only-static-ascendc --rounds 1 --timing-rounds 6 --pairs 9 --initial-warmup 100 --warmup 10 --output "$run_dir/kv${kv}.json" 2>&1 | tee "$run_dir/kv${kv}.log"
  python -u -m benchmarks.mha.summarize_fused_output "$run_dir"
done
