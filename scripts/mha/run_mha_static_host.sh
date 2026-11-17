#!/usr/bin/env bash
set -euo pipefail
printf "[FGW] initializing run_mha_static_host.sh\n"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DISABLE_CACHE=1
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_static_host.XXXXXXXX)"
publish() {
  printf 'run_directory=%s\n' "$run_dir" > mha_static_host_paste.txt
  for phase in smoke full; do
    if [ -f "$run_dir/${phase}_paste.txt" ]; then
      printf '\nphase=%s\n' "$phase" >> mha_static_host_paste.txt
      cat "$run_dir/${phase}_paste.txt" >> mha_static_host_paste.txt
    fi
  done
  for guard_log in "$run_dir"/guard*.jsonl; do
    if [ -f "$guard_log" ]; then
      printf "\nguard=%s\n" "$guard_log" >> mha_static_host_paste.txt
      tail -n 1 "$guard_log" >> mha_static_host_paste.txt
    fi
  done
  cat mha_static_host_paste.txt
}
trap publish EXIT
python -m unittest tests.mha.test_static_schedule -v 2>&1 | tee "$run_dir/host_tests.log"
python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_smoke.jsonl" -- python -u -m benchmarks.mha.bench_static_host --smoke --output "$run_dir/smoke.json" 2>&1 | tee "$run_dir/smoke.log"
python -u -m benchmarks.mha.resource_guard --log "$run_dir/guard_full.jsonl" -- python -u -m benchmarks.mha.bench_static_host --pairs 7 --warmup 5 \
  --output "$run_dir/full.json" 2>&1 | tee "$run_dir/full.log"
