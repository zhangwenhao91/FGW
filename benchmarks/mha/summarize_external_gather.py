"""Small pasteable report, including compile/accuracy failures."""
import json
from pathlib import Path


def main():
    lines = []
    for name in ("mha_external_gather_smoke_summary.json", "mha_external_gather_summary.json"):
        path = Path(name)
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        lines.append(f"{name}: status={report['status']} phase={report['phase']} active={report.get('active_case')} variant={report.get('active_variant')}")
        if report.get("error"):
            lines.append(report["error"])
        lines.append("GROUPS: " + json.dumps(report.get("groups", {})))
        for row in report.get("summary", []):
            times = {k: round(v['us'], 3) for k, v in row['metrics'].items()}
            lines.append(f"{row['key']} us={times} gather_vs_direct={row.get('speedup_gather_over_direct')} gather_vs_AscendC={row.get('speedup_gather_over_ascendc')} config={row.get('gather_configuration')}")
    text = "\n".join(lines) + "\n"
    Path("mha_external_gather_paste.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
