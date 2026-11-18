#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_merge_grouped.XXXXXXXX)"
publish() {
  python -u -m benchmarks.mha.summarize_merge_grouped "$run_dir"
  printf 'run_directory=%s\n' "$run_dir"
  cat mha_merge_grouped_paste.txt
}
trap publish EXIT
python -m unittest tests.mha.test_static_schedule -v 2>&1 | tee "$run_dir/host_tests.log"
completed=0
for kv in 4096 8192; do
  for budget in 28 56; do
    key="kv${kv}_p${budget}"
    printf '[FGW merge] starting %d/4 %s\n' "$((completed + 1))" "$key"
    python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_${key}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --shape 32768 "$kv" --plan-policy stream --merge-policy grouped --merge-programs "$budget" --merge-rows 8 --only-static-ascendc --rounds 1 --pairs 3 --warmup 5 --output "$run_dir/$key.json" 2>&1 | tee "$run_dir/$key.log"
    completed=$((completed + 1))
    printf '[FGW merge] completed %d/4 %s\n' "$completed" "$key"
    python -u -m benchmarks.mha.summarize_merge_grouped "$run_dir"
  done
done
