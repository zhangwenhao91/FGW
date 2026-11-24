#!/usr/bin/env bash
# Run comparable instruction simulations on an isolated SFA OPP and TileLang.
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "Usage: bash $0 reports/dsa/sfa-isolated.XXXX/opp" >&2
    exit 2
fi
fgw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
opp="$(cd "$1" && pwd -P)"
case "$opp" in
    "$fgw"/reports/dsa/sfa-isolated.*/opp) ;;
    *) echo "Pass the verified isolated OPP directory." >&2; exit 2 ;;
esac
command -v msprof >/dev/null
cd "$fgw"
mkdir -p reports/dsa
report="$(mktemp -d "$fgw/reports/dsa/instructions.XXXXXXXX")"
printf 'report=%s\n' "$report"
python_bin="$(command -v python)"
soc="${FGW_SIM_SOC:-Ascend950PR_9579}"
for impl in sfa tilelang; do
    trace="$report/$impl"
    mkdir "$trace"
    kernel=SparseFlashAttention
    extra=()
    if [[ "$impl" == tilelang ]]; then
        kernel=forward_kernel
        extra=(--source "$report/tilelang.asc")
    fi
    printf 'Simulating %s (Q16 KV256 topk128, core 0)\n' "$impl"
    ASCEND_OPP_PATH="$opp" msprof op simulator --soc-version "$soc" \
        --core-id 0 --kernel-name "$kernel" --launch-count 1 --timeout 8 \
        --output "$trace" "$python_bin" -u -m benchmarks.dsa_instruction_driver "$impl" "${extra[@]}" \
        >"$report/$impl.log" 2>&1 || true
    tail -n 18 "$report/$impl.log"
    if grep -q "launch_completed=$impl" "$report/$impl.log" &&
       ! grep -qi 'timeout has reached' "$report/$impl.log"; then
        echo "simulation_driver_completed=$impl"
    else
        echo "simulation_driver_not_confirmed=$impl (any trace may be partial)"
    fi
    source_opt=()
    if [[ "$impl" == tilelang && -f "$report/tilelang.asc" ]]; then
        source_opt=(--tilelang-source "$report/tilelang.asc")
    fi
    python -m benchmarks.dsa_instruction_summary "$trace" "${source_opt[@]}" \
        | tee "$report/$impl.summary.json"
done
