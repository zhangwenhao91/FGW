"""Compact two-target static scheduling/merge scan; retain incomplete status."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    expected = [f'kv{kv}_{variant}' for kv in (4096, 8192)
                for variant in ('cost', 'p28r8', 'p28r16', 'p28r32', 'p27r32')]
    results, problems = [], []
    for key in expected:
        path = directory / (key + '.json')
        if not path.exists():
            problems.append(key + ': missing')
            continue
        try:
            report = json.loads(path.read_text())
            if report['status'] != 'passed' or len(report['summary']) != 1:
                problems.append(f"{key}: {report['status']} phase={report.get('phase')} active={report.get('active')}")
                continue
            row = report['summary'][0]
            results.append(dict(key=key, settings=report['settings'], result=row))
        except (ValueError, KeyError) as exc:
            problems.append(f'{key}: {exc}')
    lines = [f"status={'passed' if not problems else 'incomplete_or_error'} completed={len(results)}/10",
             'B1H1 BF16 D128 Q32768, page128, noncausal; 1 build, 3 paired samples, 5 immediate warmups, 2 accuracy seeds.',
             'variant KV programs parts merge_rows static_us AscendC_us speedup local_us merge_us']
    for item in results:
        row = item['result']; t = row['median_us']; p = row['plan']
        lines.append(f"{item['key']} {p['KV']} {p['programs']} {p['partial_count']} "
                     f"{item['settings']['merge_rows']} {t['static']:.3f} {t['ascendc']:.3f} "
                     f"{row['static_vs_ascendc']['speedup']:.4f} {t['static/local']:.3f} "
                     f"{t.get('static/merge', 0):.3f}")
    lines.extend(problems)
    output = directory.parent / 'mha_static_targets_paste.txt'
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    (directory / 'summary.json').write_text(json.dumps(dict(results=results, problems=problems), indent=2))
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
