#!/usr/bin/env bash
# Compare installed and rebuilt SFA in separate processes without installing either.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash $0 /path/to/rebuilt/sparse_flash_attention [npu:0]" >&2
    exit 2
fi

fgw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
rebuilt="$(cd "$1" && pwd -P)"
device="${2:-npu:0}"
installed="$(realpath "${ASCEND_OPP_PATH:?Source the target CANN set_env.sh first}")"
relative=built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/sparse_flash_attention
name=SparseFlashAttention_9ec0bacf213f2b2dce3ba70149a4903b
for ext in o json; do
    test -f "$installed/$relative/$name.$ext" && test -f "$rebuilt/$name.$ext" || {
        echo "Missing installed or rebuilt BF16 file: $name.$ext" >&2
        exit 2
    }
done

cd "$fgw"
mkdir -p reports/dsa
report="$(mktemp -d "$fgw/reports/dsa/sfa-isolated.XXXXXXXX")"
isolated="$report/opp"
printf 'report=%s\n' "$report"
echo "Copying OPP to an independent directory (copy-on-write when supported)."
mkdir "$isolated"
cp -a --reflink=auto -- "$installed/." "$isolated/"
target="$(realpath "$isolated/$relative")"
if [[ "$target" != "$isolated/"* ]]; then
    echo "Isolated destination resolves outside the copied OPP: $target" >&2
    exit 2
fi
for ext in o json; do
    cp --remove-destination -- "$rebuilt/$name.$ext" "$target/$name.$ext"
done
sha256sum "$installed/$relative/$name.o" "$target/$name.o" | tee "$report/objects.sha256"

for variant in installed rebuilt; do
    opp="$installed"
    if [[ "$variant" == rebuilt ]]; then opp="$isolated"; fi
    if command -v strace >/dev/null; then
        echo "Auditing device object opens: $variant"
        env ASCEND_OPP_PATH="$opp" strace -f -yy -e trace=open,openat,openat2 \
            -o "$report/$variant.open.log" python -u -m benchmarks.dsa_dense_sfa \
            --case native-sfa-long --device "$device" --verify-only \
            2>&1 | tee "$report/$variant.verify.log"
        grep -F "$name.o" "$report/$variant.open.log" | grep -E '= [0-9]+' \
            > "$report/$variant.loaded.txt" || true
        cat "$report/$variant.loaded.txt"
    else
        echo "strace unavailable: object loading cannot be verified by this script."
    fi
    echo "Measuring and profiling $variant"
    ASCEND_OPP_PATH="$opp" python -u -m benchmarks.dsa_dense_sfa \
        --case native-sfa-long --device "$device" \
        --output "$report/$variant.json" --profile-dir "$report/$variant.profile" \
        2>&1 | tee "$report/$variant.log"
done
python - "$report" "$installed" "$isolated" "$name" <<'PY'
import json
import sys
from pathlib import Path

directory, installed, isolated, name = sys.argv[1:]
root = Path(directory)
result = {"variants": {}}
for variant, opp in (("installed", installed), ("rebuilt", isolated)):
    timing = json.loads((root / f"{variant}.json").read_text())
    evidence = root / f"{variant}.loaded.txt"
    hits = evidence.read_text().splitlines() if evidence.exists() else []
    selected = [line for line in hits if f"{opp}/" in line and f"{name}.o" in line]
    variant_root = root / f"{variant}.profile"
    profiles = sorted(variant_root.rglob("summary.json"))
    result["variants"][variant] = {
        "expected_object_open_observed": bool(selected),
        "matching_open_examples": selected[:2],
        "event_median_us": timing["median_us"]["native_sfa"],
        "accuracy": timing["accuracy"]["native_sfa"],
        "profile": json.loads(profiles[0].read_text()) if len(profiles) == 1 else None,
    }
result["rebuilt_over_installed_event_latency"] = (
    result["variants"]["rebuilt"]["event_median_us"]
    / result["variants"]["installed"]["event_median_us"]
)
text = json.dumps(result, indent=2)
(root / "comparison.json").write_text(text + "\n")
print(text)
PY
