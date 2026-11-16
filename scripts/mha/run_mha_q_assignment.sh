#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
run_dir="$(mktemp -d mha_q_assignment.XXXXXXXX)"
publish() {
  printf 'run_directory=%s\n' "$run_dir" > mha_q_assignment_paste.txt
  for phase in smoke full boundary; do
    if [ -f "$run_dir/${phase}_paste.txt" ]; then
      printf '\nphase=%s\n' "$phase" >> mha_q_assignment_paste.txt
      cat "$run_dir/${phase}_paste.txt" >> mha_q_assignment_paste.txt
    fi
  done
  cat mha_q_assignment_paste.txt
}
trap publish EXIT
python -u -m benchmarks.mha.bench_q_assignment --smoke --rounds 2 --pairs 7 --warmup 5 \
  --output "$run_dir/smoke.json" 2>&1 | tee "$run_dir/smoke.log"
python -u -m benchmarks.mha.bench_q_assignment --rounds 6 --pairs 51 --warmup 30 \
  --output "$run_dir/full.json" 2>&1 | tee "$run_dir/full.log"
python -u -m benchmarks.mha.bench_q_assignment --boundary --rounds 6 --pairs 51 --warmup 30 \
  --output "$run_dir/boundary.json" 2>&1 | tee "$run_dir/boundary.log"
