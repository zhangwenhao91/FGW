"""Compact clean-v7/FIA grid report, preserving failures and incomplete cases."""
import argparse
import csv
import json
from pathlib import Path

QS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
KVS = (512, 1024, 2048, 4096, 8192, 16384, 24576, 32768)


def summarize(path, output):
    report = json.loads(path.read_text(encoding="utf-8"))
    balanced = report.get("settings", {}).get("balanced_grid", False)
    qs = (3584,7168,17920,28672) if balanced else QS
    kvs = (8192,16384,32768) if balanced else KVS
    expected = len(qs)*len(kvs)
    rows = report.get("summary", [])
    lines = [f"status={report['status']} phase={report['phase']} correctness={report.get('correctness')} "
             f"completed={len(rows)}/{expected} active={report.get('active')}",
             "B1H1 BF16 D128 page128 noncausal; speedup=AscendC_us/TL_us; two TL builds, two rounds.",
             "MATRIX: rows=Q, columns=KV; -=incomplete"]
    by_key = {r["key"]: r for r in rows}
    lines.append("Q/KV " + " ".join(map(str, kvs)))
    for q in qs:
        cells = [f"{by_key[f'q{q}_kv{kv}']['baseline_vs_ascendc']['speedup']:.4f}"
                 if f"q{q}_kv{kv}" in by_key else "-" for kv in kvs]
        lines.append(str(q) + " " + " ".join(cells))
    lines.append("DETAIL: Q KV split TL_us AscendC_us speedup round_min round_max TL_builds")
    for r in rows:
        ratio = r["baseline_vs_ascendc"]
        lines.append(f"{r['key']} S{r['split']} {r['median_us']['baseline']:.3f} "
                     f"{r['median_us']['ascendc']:.3f} {ratio['speedup']:.4f} "
                     f"{ratio['min_round']:.4f} {ratio['max_round']:.4f} "
                     f"{r['fresh_compile_count']['baseline']}")
        if balanced:
            lines.append("  BALANCE " + json.dumps(r["balance"]) + " TFLOPS " + json.dumps(r["useful_tflops"]))
    if report.get("groups"):
        lines.append("GROUPS: " + json.dumps(report["groups"]))
    if report.get("error"):
        lines.append(report["error"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Q/KV", *kvs])
        for q in qs:
            writer.writerow([q, *[by_key[f"q{q}_kv{kv}"]["baseline_vs_ascendc"]["speedup"]
                                  if f"q{q}_kv{kv}" in by_key else "" for kv in kvs]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.report, args.output)
