"""Summarize independently supervised static-host grid cases, including failures."""
import argparse
import json
from pathlib import Path

from benchmarks.mha.summarize_clean_grid import QS, KVS


def summarize(directory):
    rows = []
    failures = []
    policies = set()
    for kv in KVS:
        for q in QS:
            key = f"q{q}_kv{kv}"
            path = directory / f"{key}.json"
            if not path.exists():
                continue
            try:
                report = json.loads(path.read_text())
                if report['status'] == 'passed' and len(report['summary']) == 1:
                    rows.append(report['summary'][0])
                    settings = report.get('settings', {})
                    policies.add((settings.get('rounds', 2), settings.get('pairs', 7), settings.get('warmup', 5)))
                else:
                    failures.append(f"{key}: {report['status']} phase={report.get('phase')} active={report.get('active')}")
            except (ValueError, KeyError) as exc:
                failures.append(f"{key}: unreadable report {exc}")
    passed = len(rows) == len(QS) * len(KVS) and not failures
    lines = [f"status={'passed' if passed else 'incomplete_or_error'} completed={len(rows)}/80",
             'B1H1 BF16 D128 page128 noncausal; 2 correctness seeds; timing policy (compile_rounds, pairs, immediate_warmups)=' + str(sorted(policies)),
             'Full-call Event timing excludes compilation and one-time host plan/H2D preparation.',
             'Q KV v7_us static_us AscendC_us v7/static AscendC/static round_AC_min round_AC_max parts merge_tiles builds']
    for row in rows:
        t, p = row['median_us'], row['plan']
        ac = row['static_vs_ascendc']
        lines.append(f"{p['Q']} {p['KV']} {t['default']:.3f} {t['static']:.3f} {t['ascendc']:.3f} "
                     f"{row['static_vs_default']['speedup']:.4f} {ac['speedup']:.4f} "
                     f"{ac['min_round']:.4f} {ac['max_round']:.4f} "
                     f"{p['partial_count']} {p['merge_q_tiles']} {row['static_builds']}")
    lines.extend(failures)
    output = directory.parent / 'mha_static_host_grid_paste.txt'
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    (directory / 'summary.json').write_text(json.dumps(dict(status='passed' if passed else 'incomplete_or_error',
                                                          completed=len(rows), expected=80, failures=failures,
                                                          summary=rows), indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
