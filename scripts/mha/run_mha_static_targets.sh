#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_static_targets.XXXXXXXX)"
publish() {
  python -u -m benchmarks.mha.summarize_static_targets "$run_dir"
  printf 'run_directory=%s\n' "$run_dir"
  cat mha_static_targets_paste.txt
}
trap publish EXIT
completed=0
started=$SECONDS
for kv in 4096 8192; do
  for variant in cost p28r8 p28r16 p28r32 p27r32; do
    options=(--plan-policy balanced --programs 28 --merge-rows 8)
    case "$variant" in
      cost) options=(--plan-policy cost --merge-rows 8) ;;
      p28r16) options=(--plan-policy balanced --programs 28 --merge-rows 16) ;;
      p28r32) options=(--plan-policy balanced --programs 28 --merge-rows 32) ;;
      p27r32) options=(--plan-policy balanced --programs 27 --merge-rows 32) ;;
    esac
    key="kv${kv}_${variant}"
    printf '[FGW targets] starting %d/10 %s elapsed=%ss\n' "$((completed + 1))" "$key" "$((SECONDS - started))"
    python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_${key}.jsonl" -- python -u -m benchmarks.mha.bench_static_host --shape 32768 "$kv" "${options[@]}" --rounds 1 --pairs 3 --warmup 5 --output "$run_dir/$key.json" 2>&1 | tee "$run_dir/$key.log"
    completed=$((completed + 1))
    printf '[FGW targets] completed %d/10 %s\n' "$completed" "$key"
    python -u -m benchmarks.mha.summarize_static_targets "$run_dir"
  done
done
