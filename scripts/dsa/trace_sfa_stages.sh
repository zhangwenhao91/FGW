#!/usr/bin/env bash
# Run comparable instruction simulations on an isolated SFA OPP and TileLang.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash $0 reports/dsa/sfa-isolated.XXXX/opp [32|128]" >&2
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
tile="${2:-32}"
case "$tile" in
    32) shape=Q16-KV256-topk128 ;;
    128) shape=Q16-KV512-topk256 ;;
    *) echo "Expected tile size 32 or 128." >&2; exit 2 ;;
esac
implementations=(sfa tilelang)
if [[ "$tile" == 128 ]]; then
    implementations+=(tilelang-packed)
fi
for impl in "${implementations[@]}"; do
    trace="$report/$impl"
    mkdir "$trace"
    kernel=SparseFlashAttention
    extra=()
    if [[ "$impl" != sfa ]]; then
        kernel=forward_kernel
        extra=(--source "$report/$impl.asc")
    fi
    printf 'Simulating %s (%s, TileLang KV tile %s, core 0)\n' "$impl" "$shape" "$tile"
    ASCEND_OPP_PATH="$opp" msprof op simulator --soc-version "$soc" \
        --core-id 0 --kernel-name "$kernel" --launch-count 1 --timeout 8 \
        --output "$trace" "$python_bin" -u -m benchmarks.dsa_instruction_driver "$impl" --kv-tile "$tile" "${extra[@]}" \
        >"$report/$impl.log" 2>&1 || true
    tail -n 18 "$report/$impl.log"
    if grep -q "launch_completed=$impl" "$report/$impl.log" &&
       ! grep -qi 'timeout has reached' "$report/$impl.log"; then
        echo "simulation_driver_completed=$impl"
    else
        echo "simulation_driver_not_confirmed=$impl (any trace may be partial)"
    fi
    source_opt=()
    if [[ "$impl" != sfa && -f "$report/$impl.asc" ]]; then
        source_opt=(--tilelang-source "$report/$impl.asc")
    fi
    python -m benchmarks.dsa_instruction_summary "$trace" "${source_opt[@]}" \
        | tee "$report/$impl.summary.json"
    python -m benchmarks.dsa_vf_calls "$trace" "${source_opt[@]}"         > "$report/$impl.vf-calls.json"
done
