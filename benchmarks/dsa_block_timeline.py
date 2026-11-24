"""Compare matched KV128 QK, probability DMA, and PV events on simulator core 0."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def source_markers(path):
    lines = path.read_text().splitlines()

    def unique(pattern):
        matches = [i for i, line in enumerate(lines, 1) if re.search(pattern, line)]
        if len(matches) != 1:
            raise ValueError(
                f"{path}: expected one source marker for {pattern}, got {matches}"
            )
        return matches[0]

    qk = unique(r"asc_copy_l0c2ub\(.*\), 128, 64, 128, 64,")
    pv = unique(r"asc_mmad\(.*\), 64, 32, 512,.*piece == 0")
    prob = {}
    for sid in (0, 1):
        notify = [
            i
            for i, line in enumerate(lines)
            if "asc_sync_intra_arrive(PIPE_MTE3, 4)" in line
        ]
        if len(notify) != 2 or "asc_copy_ub2l1(" not in lines[notify[sid] - 1]:
            raise ValueError(f"{path}: probability DMA marker missing")
        prob[sid] = notify[sid]  # preceding source line, converted to one-based
    return qk, pv, prob


def events_at(root, core, markers, *, inspect=False):
    paths = sorted(root.rglob(f"core0.{core}/trace.json"))
    if len(paths) != 1:
        raise ValueError(f"{root}: expected one trace for {core}, got {len(paths)}")
    data = json.loads(paths[0].read_text())
    groups = {line: set() for line in markers}
    details = {line: Counter() for line in markers}
    for event in data if isinstance(data, list) else data["traceEvents"]:
        if event.get("ph") != "X" or event.get("dur", 0) <= 0:
            continue
        args = event.get("args", {})
        detail = args.get("detail", {})
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except ValueError:
                continue
        if isinstance(detail, dict) and any(
            detail.get(key) for key in ("is_sync", "is_wait", "is_duration_excluded")
        ):
            continue
        source_lines = {
            int(line) for line in re.findall(r"\.asc:(\d+)", str(args.get("code", "")))
        }
        for line in source_lines.intersection(groups):
            if inspect:
                details[line][
                    (str(event.get("name", "")), str(args.get("code", ""))[:350])
                ] += 1
            groups[line].add(
                (event["ts"], event["ts"] + event["dur"], event.get("name", ""))
            )
    if inspect:
        return {
            line: [
                {"event": name, "source": code, "count": count}
                for (name, code), count in counts.most_common()
            ]
            for line, counts in details.items()
        }
    return {line: sorted(items) for line, items in groups.items()}


def summarize(root, impl):
    qk_line, pv_line, prob_lines = source_markers(root / f"{impl}.asc")
    trace = root / impl
    cube = events_at(trace, "cubecore0", (qk_line, pv_line))
    qk, pv = cube[qk_line], cube[pv_line]
    prob = [
        events_at(trace, f"veccore{sid}", (line,))[line]
        for sid, line in prob_lines.items()
    ]
    counts = [len(qk), len(prob[0]), len(prob[1]), len(pv)]
    if not counts[0] or counts[0] % 2 or counts != [len(qk)] * 3 + [4 * len(qk)]:
        raise ValueError(f"{impl}: unexpected QK/Vec0 DMA/Vec1 DMA/PV counts {counts}")
    rows = []
    origin = qk[0][1]
    for i, score in enumerate(qk):
        dma = [events[i] for events in prob]
        ready = max(event[1] for event in dma)
        start = pv[4 * i][0]
        if not score[1] <= min(event[0] for event in dma) <= ready <= start:
            raise ValueError(f"{impl}: timestamps for block {i} fail causal ordering")
        rows.append(
            {
                "task": i // 2,
                "kv_block": i % 2,
                "qk_end_us": round(score[1] - origin, 5),
                "probability_l1_ready_us": round(ready - origin, 5),
                "pv_start_us": round(start - origin, 5),
                "qk_to_probability_us": round(ready - score[1], 5),
                "probability_to_pv_us": round(start - ready, 5),
                "qk_to_pv_us": round(start - score[1], 5),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--inspect-markers", action="store_true")
    args = parser.parse_args()
    try:
        if args.inspect_markers:
            report = {}
            for impl in ("tilelang", "tilelang-packed"):
                qk, pv, prob = source_markers(args.report / f"{impl}.asc")
                trace = args.report / impl
                report[impl] = {
                    "qk_and_pv_source_lines": [qk, pv],
                    "cube": events_at(trace, "cubecore0", (qk, pv), inspect=True),
                    "probability_dma": {
                        f"vec{sid}": events_at(
                            trace, f"veccore{sid}", (line,), inspect=True
                        )
                        for sid, line in prob.items()
                    },
                }
            print(json.dumps(report, indent=2))
            return
        baseline = summarize(args.report, "tilelang")
        packed = summarize(args.report, "tilelang-packed")
        paired = list(zip(baseline, packed, strict=True))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(
        json.dumps(
            {
                "timestamp_origin": "first QK writeback end in each trace",
                "tilelang": baseline,
                "tilelang_packed": packed,
                "baseline_minus_packed_us": [
                    {
                        key: round(old[key] - new[key], 5)
                        for key in old
                        if key.endswith("_us")
                    }
                    for old, new in paired
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
