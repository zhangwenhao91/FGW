#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$repo_root/reports/mha/latest"
cd -- "$repo_root/reports/mha/latest"
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --suite cube \
  --patterns full --dtypes bfloat16 --causal noncausal --output mha_persistent28_control_accuracy.json \
  2>&1 | tee mha_persistent28_control_accuracy.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --suite cube --persistent-programs 28 \
  --patterns full --dtypes bfloat16 --causal noncausal --output mha_persistent28_accuracy.json \
  2>&1 | tee mha_persistent28_accuracy.log
python -u -m tests.mha.test_mha_paged_accuracy --impl v7 --suite cube --persistent-programs 28 \
  --cases cube_decode512 cube_decode2048 cube_direct_q1 --patterns full --dtypes bfloat16 \
  --causal causal --output mha_persistent28_causal.json 2>&1 | tee mha_persistent28_causal.log
python -u -m benchmarks.mha.bench_v7_ascendc --compare-persistent --fia-api v2 --kv 2048 \
  --q-lengths 1 129 4096 --rounds 1 --warmup 5 --repeat 11 --output mha_persistent28_smoke.json \
  2>&1 | tee mha_persistent28_smoke.log
failed=0
for kv in 2048 8192; do
  name="mha_persistent28_kv${kv}"
  if ! python -u -m benchmarks.mha.bench_v7_ascendc --compare-persistent --fia-api v2 --kv "$kv" \
    --q-lengths 1 16 128 129 512 1024 4096 8192 16384 32768 \
    --rounds 4 --warmup 20 --repeat 51 --output "${name}.json" 2>&1 | tee "${name}.log"; then
    failed=1
  fi
done
if [ "$failed" = 0 ]; then
  python -u -m benchmarks.mha.profile_v7_ascendc_msprof --compare-persistent --rounds 3 \
    --warmup 20 --repeat 30 --event-repeat 51 --output mha_persistent28_msprof.json \
    2>&1 | tee mha_persistent28_msprof.log
fi
exit "$failed"
