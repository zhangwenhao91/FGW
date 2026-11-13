"""Export compact general-model accuracy/selection results for terminal pasting."""
import argparse
import datetime
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", help="Only include reports started at/after this ISO timestamp")
    parser.add_argument("--directory", default=".")
    parser.add_argument("--output", default="mha_cube_model_status.txt")
    args = parser.parse_args()
    since = datetime.datetime.fromisoformat(args.since) if args.since else None
    lines = []
    for path in sorted(Path(args.directory).glob("mha_cube_model_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        started = r.get("started_utc")
        if since and (not started or datetime.datetime.fromisoformat(started) < since):
            continue
        lines.extend(["FILE: "+path.name, "started_utc: "+str(started)])
        for key in ("status", "phase", "passed", "total_checks", "correctness", "active_case", "active_split"):
            if key in r:
                lines.append(f"{key}: {r[key]}")
        error = r.get("error") or next((v.get("error") for v in r.get("results", [])
                                         if v.get("status") in ("failed", "error") and v.get("error")), None)
        if error:
            lines.append("ERROR:\n"+error[-8000:])
        if r.get("groups"):
            lines.append("GROUPS: "+json.dumps(r["groups"], ensure_ascii=False))
        for s in r.get("summary", []):
            model = s.get("model_splits")
            if model is None:
                continue
            timings = s["variants"]
            lines.append(f"{s['key']} auto={s['auto_splits']} model={model} best={s['best_split']} near2%={s['near_best_2pct']} "
                         f"auto_us={timings[str(s['auto_splits'])]['us']:.3f} model_us={timings[str(model)]['us']:.3f} "
                         f"best_us={timings[str(s['best_split'])]['us']:.3f} model_regret={s['model_regret_percent']:.2f}% "
                         f"model_vs_auto={s['model_speedup_vs_auto']:.4f}x")
        lines.append("")
    output = Path(args.directory)/args.output
    output.write_text("\n".join(lines) if lines else "No reports from this run found.\n", encoding="utf-8")
    print(f"Paste summary: {output}", flush=True)


if __name__ == "__main__":
    main()
