"""Same-run grouped/fixed R8/AscendC comparison for the two targets."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    lines = ['Q=32768 B1H1 BF16 D128; shared local kernel; 1 build round, 3 rotated triples, 5 immediate warmups, 2 accuracy seeds.',
             'KV merge_budget grouped_us fixed_R8_us AscendC_us fixed/grouped AC/grouped programs max_chunks']
    complete = 0
    for kv in (4096, 8192):
        for budget in (28, 56):
            key = f'kv{kv}_p{budget}'
            path = directory / (key + '.json')
            if not path.exists():
                lines.append(key + ': missing')
                continue
            try:
                report = json.loads(path.read_text())
                if report['status'] != 'passed' or len(report['summary']) != 1:
                    lines.append(f"{key}: {report['status']} phase={report.get('phase')} active={report.get('active')}")
                    continue
                row = report['summary'][0]; t = row['median_us']; assignment = row['merge_assignment']
                lines.append(f"{kv} {budget} {t['static']:.3f} {t['fixed_r8']:.3f} {t['ascendc']:.3f} "
                             f"{row['static_vs_fixed_r8']['speedup']:.4f} {row['static_vs_ascendc']['speedup']:.4f} "
                             f"{assignment['programs']} {assignment['max_chunks']}")
                for measurement in report.get('rounds', []):
                    lines.append('SAMPLES_us ' + key + ' ' + json.dumps(measurement['samples_us'], sort_keys=True))
                complete += 1
            except (ValueError, KeyError) as exc:
                lines.append(f'{key}: invalid report {exc}')
    lines.insert(0, f"status={'passed' if complete == 4 else 'incomplete_or_error'} completed={complete}/4")
    output = directory.parent / 'mha_merge_grouped_paste.txt'
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
