#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Independent Python processes preserve failures and isolate compiler/runtime state.
# Counterbalance candidate order across two sweeps; do not choose a winner yet.
failed=0
for sweep in 1 2; do
  if [ "$sweep" = 1 ]; then candidates="1 2 3"; else candidates="3 2 1"; fi
  for depth in $candidates; do
    name="mha_pipeline_s${depth}_v${depth}_run${sweep}"
    if python -u profile_mha_msprof.py --pipeline-stages "$depth" --buffer-versions "$depth" \
      --rounds 2 --output "${name}.json" 2>&1 | tee "${name}.log"; then
      printf 'Completed %s\n' "$name"
    else
      printf 'Failed %s; report/log retained, continuing other candidates\n' "$name"
      failed=1
    fi
  done
done
exit "$failed"
