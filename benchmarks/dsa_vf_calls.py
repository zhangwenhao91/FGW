"""Count actual simulator VF calls and measure attributed instructions per call."""

import argparse
import bisect
import collections
import json
import re
import statistics
from pathlib import Path

from benchmarks.dsa_instruction_summary import phase, tl_spans, union


def callsite_phase(code, source):
    if not source:
        return phase(code, [])
    stages = ("softmax", "pv_update", "output_normalize")
    found = set()
    for line in re.findall(r"\.asc:(\d+)", code):
        index = int(line) - 1
        if 0 <= index < len(source):
            match = re.search(r"forward_kernel_simd_vf_([012])(?:_1)?\(", source[index])
            if match:
                found.add(stages[int(match[1])])
            if "ascend_nd2nz_scatter" in source[index]:
                found.add("kv_nd2nz")
    return next(iter(found)) if len(found) == 1 else "unattributed"


def summarize(path, source, spans):
    data = json.loads(path.read_text())
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    launches, instructions = [], []
    for event in events:
        if event.get("ph") != "X" or not all(k in event for k in ("ts", "dur")):
            continue
        name = event.get("name", "")
        args = event.get("args", {})
        code = str(args.get("code", ""))
        start = event["ts"]
        end = start + event["dur"]
        if name == "VF":
            launches.append((start, end, code, str(args.get("pc_addr", ""))))
        elif name.startswith("RV_"):
            detail = args.get("detail", {})
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except ValueError:
                    continue
            if any(
                detail.get(key)
                for key in ("is_sync", "is_wait", "is_duration_excluded")
            ):
                continue
            instructions.append((start, end, name, phase(code, spans)))
    instructions.sort()
    starts = [item[0] for item in instructions]
    counts = collections.Counter()
    durations = collections.defaultdict(list)
    reductions = collections.defaultdict(collections.Counter)
    eps = 1e-6
    pc_counts = collections.Counter()
    pc_labels = collections.defaultdict(collections.Counter)
    for start, end, code, pc in launches:
        pc_counts[pc] += 1
        lo = bisect.bisect_left(starts, start - eps)
        hi = bisect.bisect_right(starts, end + eps)
        children = [item for item in instructions[lo:hi] if item[1] <= end + eps]
        known = {item[3] for item in children} - {"unattributed"}
        direct = callsite_phase(code, source)
        if direct != "unattributed":
            label = direct
        elif len(known) == 1:
            label = next(iter(known))
        else:
            label = "mixed" if known else "unattributed"
        counts[label] += 1
        pc_labels[pc][label] += 1
        attributed = [item for item in children if item[3] == label]
        if attributed and label not in ("mixed", "unattributed"):
            durations[label].append(union([(item[0], item[1]) for item in attributed]))
            reductions[label][sum(item[2] == "RV_VCMAX" for item in attributed)] += 1
    return {
        "vf_launches": len(launches),
        "launch_pc_groups": {
            pc or "(missing PC)": {"count": count, "labels": dict(pc_labels[pc])}
            for pc, count in sorted(pc_counts.items())
        },
        "calls": {
            label: {
                "count": count,
                "with_attributed_rv_instructions": len(durations[label]),
                "median_attributed_active_us_per_call": (
                    statistics.median(durations[label]) if durations[label] else None
                ),
                "vcmax_events_per_attributed_call_histogram": dict(reductions[label]),
            }
            for label, count in sorted(counts.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--tilelang-source", type=Path)
    args = parser.parse_args()
    source = (
        args.tilelang_source.read_text().splitlines() if args.tilelang_source else []
    )
    spans = tl_spans(args.tilelang_source) if args.tilelang_source else []
    paths = sorted(
        path for path in args.trace.rglob("trace.json") if "veccore" in path.parent.name
    )
    print(
        json.dumps(
            {
                "trace_files": len(paths),
                "tilelang_query_head_rows_per_compute_vf": 32 if source else None,
                "vector_cores": {
                    str(path.relative_to(args.trace)): summarize(path, source, spans)
                    for path in paths
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
