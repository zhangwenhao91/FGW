"""Summarize source-attributed, non-sync instruction intervals per Vector core."""

import argparse
import collections
import json
import re
from pathlib import Path


def union(intervals):
    total = end = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return round(total, 5)


def tl_spans(source):
    lines = source.read_text().splitlines()
    spans = []
    names = ("softmax", "pv_update", "output_normalize")
    for start, line in enumerate(lines, 1):
        if match := re.search(r"__simd_vf__.*simd_vf_([012])(?:_1)?\(", line):
            stop = next(
                i for i in range(start + 1, len(lines) + 1) if lines[i - 1] == "}"
            )
            spans.append((start, stop, names[int(match[1])]))
    return spans


def phase(code, spans):
    if spans:
        matches = {
            name
            for line in re.findall(r"\.asc:(\d+)", code)
            for start, stop, name in spans
            if start <= int(line) <= stop
        }
        return next(iter(matches)) if len(matches) == 1 else "unattributed"
    code = code.lower()
    if "flashupdatelast" in code or "lastdiv" in code:
        return "pv_last_and_output"
    if "flashupdate" in code:
        return "pv_update_including_possible_final"
    if "softmaxflash" in code or "processvec1" in code:
        return "softmax"
    return "unattributed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--tilelang-source", type=Path)
    args = parser.parse_args()
    spans = tl_spans(args.tilelang_source) if args.tilelang_source else []
    if args.tilelang_source and not spans:
        parser.error("no known SIMD function ranges found in TileLang source")
    result = {}
    for path in sorted(args.trace.rglob("trace.json")):
        if "veccore" not in path.parent.name:
            continue
        trace = json.loads(path.read_text())
        groups = collections.defaultdict(list)
        for event in trace if isinstance(trace, list) else trace.get("traceEvents", []):
            if event.get("ph") != "X" or "dur" not in event:
                continue
            data = event.get("args", {})
            detail = data.get("detail", {})
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
            code = str(data.get("code", ""))
            start = event["ts"]
            groups[phase(code, spans)].append((start, start + event["dur"]))
        result[str(path.relative_to(args.trace))] = {
            stage: {"events": len(values), "active_interval_union_us": union(values)}
            for stage, values in sorted(groups.items())
        }
    print(json.dumps({"trace_files": len(result), "vector_cores": result}, indent=2))


if __name__ == "__main__":
    main()
