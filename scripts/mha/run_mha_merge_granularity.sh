#!/usr/bin/env bash
set -euo pipefail
printf "[FGW] initializing run_mha_merge_granularity.sh\n"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
resume_args=()
if [ "${1:-}" = "--resume" ]; then
  resume_args=(--resume "$(realpath -- "${2:?Missing previous full.json}")")
  shift 2
fi
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_merge_granularity.XXXXXXXX)"
publish() {
  printf 'run_directory=%s\n' "$run_dir" > mha_merge_granularity_paste.txt
  if [ -f "$run_dir/full_paste.txt" ]; then
    cat "$run_dir/full_paste.txt" >> mha_merge_granularity_paste.txt
  fi
  for guard_log in "$run_dir"/guard*.jsonl; do
    if [ -f "$guard_log" ]; then
      printf "\nguard=%s\n" "$guard_log" >> mha_merge_granularity_paste.txt
      tail -n 1 "$guard_log" >> mha_merge_granularity_paste.txt
    fi
  done
  cat mha_merge_granularity_paste.txt
}
trap publish EXIT
python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_full.jsonl" -- python -u -m benchmarks.mha.bench_split_breakdown --merge-granularity --pairs 7 --warmup 5 \
  --output "$run_dir/full.json" "${resume_args[@]}" "$@" 2>&1 | tee "$run_dir/full.log"
