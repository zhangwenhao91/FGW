#!/usr/bin/env bash
# Build only an optimized SFA kernel with debug line information. Do not install.
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: bash $0 /path/to/separate/ops-transformer-checkout" >&2
    exit 2
fi
src="$(cd "$1" && pwd -P)"
fgw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ ! -f "$src/build.sh" || ! -d "$src/attention/sparse_flash_attention" ]]; then
    echo "Need a checked-out ops-transformer source tree." >&2
    exit 2
fi
if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "$ASCEND_HOME_PATH" ]]; then
    echo "Source the intended CANN set_env.sh first." >&2
    exit 2
fi
if [[ "$src" == "$(realpath "$ASCEND_HOME_PATH")" || "$src" == "$(realpath "$ASCEND_HOME_PATH")/"* ]]; then
    echo "Use a source checkout outside the installed CANN directory." >&2
    exit 2
fi
if [[ -e "$src/build" || -e "$src/build_out" ]]; then
    echo "Use a fresh checkout so build output cannot include stale binaries." >&2
    exit 2
fi
mkdir -p "$fgw/reports/dsa"
report="$(mktemp -d "$fgw/reports/dsa/sfa-lines.XXXXXXXX")"
printf 'report=%s\n' "$report"
{
    git -C "$src" log -1 --format='source_commit=%H%nsource_date=%aI'
    printf 'cann_home=%s\n' "$ASCEND_HOME_PATH"
    for file in "$ASCEND_HOME_PATH/opp/version.info" "$ASCEND_HOME_PATH/compiler/version.info"; do
        if [[ -f "$file" ]]; then
            printf '\n%s\n' "$file"
            head -n 15 "$file"
        fi
    done
} | tee "$report/build_identity.txt"
(
    cd "$src"
    bash build.sh --opkernel --soc=ascend950 --ops=sparse_flash_attention \
        --build-type=Release --ops-compile-options=-g -j4
) 2>&1 | tee "$report/build.log"

reader="$(command -v llvm-readelf || command -v readelf)"
count=0
while IFS= read -r -d '' obj; do
    sections="$("$reader" -WS "$obj" 2>/dev/null)" || continue
    if grep -Eq '\.(z?debug_line)([[:space:]]|$)' <<<"$sections"; then
        printf '%s\n' "$obj" | tee -a "$report/line_table_objects.txt"
        count=$((count + 1))
    fi
done < <(find "$src" -type f -iname 'SparseFlashAttention_*.o' -print0)
printf 'objects_with_line_table=%d\n' "$count" | tee -a "$report/build_identity.txt"
if (( count == 0 )); then
    echo "No SFA device object with a line table found. Check $report/build.log." >&2
    exit 1
fi
