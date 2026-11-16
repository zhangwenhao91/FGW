"""Summarize a fresh B1H1 grid directory into portable JSON and pasteable tables."""
import collections
import json
import math
from pathlib import Path
import statistics
import sys

QS = (1, 4, 16, 32, 64, 127, 128, 129, 256, 512, 1024, 2048, 4096, 8192)
KVS = (512, 2048, 4096, 8192)


def summarize(directory):
    rows, errors = [], []
    for kv in KVS:
        path = directory / f"kv{kv}_summary.json"
        if not path.exists():
            errors.append(f"KV{kv}: not completed")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        settings = data["settings"]
        if settings["bh"] != ["1x1"] or settings["kv"] != kv or settings["q_lengths"] != list(QS):
            raise ValueError(f"Unexpected matrix: {path}")
        if data["status"] != "passed":
            errors.append(f"KV{kv}: {data['status']}\n{data.get('error', '')}")
            continue
        groups = collections.defaultdict(list)
        for item in data.get("comparisons", []):
            if item["candidate"] == "v7_manual_bf16" and item["baseline"] == "ascendc_fia_v2":
                if item["B"] != 1 or item["H"] != 1:
                    raise ValueError("Unexpected B/H")
                groups[item["Q"]].append(item)
        for q in QS:
            items = groups[q]
            if len(items) != settings["rounds"]:
                errors.append(f"Q{q}/KV{kv}: missing rounds")
                continue
            rows.append({"Q": q, "KV": kv, "rounds": len(items),
                         "v7_us": statistics.median(i["v7_median_us"] for i in items),
                         "ascendc_us": statistics.median(i["ascendc_median_us"] for i in items),
                         "speedup": statistics.median(i["speedup_v7_over_ascendc"] for i in items),
                         "min_round_speedup": min(i["speedup_v7_over_ascendc"] for i in items),
                         "max_round_speedup": max(i["speedup_v7_over_ascendc"] for i in items)})
    smoke = directory / "smoke_summary.json"
    if smoke.exists():
        data = json.loads(smoke.read_text(encoding="utf-8"))
        if data["status"] != "passed":
            errors.insert(0, f"smoke: {data['status']}\n{data.get('error', '')}")
    report = {"status": "passed" if not errors and len(rows) == 56 else "incomplete",
              "run_directory": str(directory), "cases": len(rows), "errors": errors,
              "policy": "B1 H1 BF16 D128 page128 noncausal; default v7 vs native FIA v2; median paired round ratios; warm-cache Event",
              "rows": rows}
    if rows:
        report["geomean_speedup"] = math.exp(statistics.mean(math.log(r["speedup"]) for r in rows))
        report["v7_faster_cases"] = sum(r["speedup"] > 1 for r in rows)
    lookup = {(r["Q"], r["KV"]): r for r in rows}
    lines = [json.dumps({k: v for k, v in report.items() if k != "rows"}),
             "Q | KV512 | KV2048 | KV4096 | KV8192 (AscendC_us/v7_us)"]
    for q in QS:
        lines.append(str(q) + " | " + " | ".join(
            f"{lookup[q, kv]['speedup']:.4f}x" if (q, kv) in lookup else "N/A" for kv in KVS))
    for r in rows:
        lines.append(f"Q={r['Q']} KV={r['KV']} v7={r['v7_us']:.3f}us AscendC={r['ascendc_us']:.3f}us speedup={r['speedup']:.4f}x rounds={r['rounds']} range={r['min_round_speedup']:.4f}..{r['max_round_speedup']:.4f}")
    return report, "\n".join(lines) + "\n"


def main():
    directory = Path(sys.argv[1])
    report, text = summarize(directory)
    for parent in (directory, Path.cwd()):
        (parent / "mha_v7_b1h1_grid_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (parent / "mha_v7_b1h1_grid_paste.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
