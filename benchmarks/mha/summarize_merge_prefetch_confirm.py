"""Confirmation timing report; retain every sample and every measurement repeat."""
import argparse
import json
from pathlib import Path
import statistics

LABELS = ('static', 'fixed_r8', 'ascendc')


def summarize(directory):
    lines = ['Q32768 B1H1 BF16 D128; one shared local build; 6 measurement repeats x 9 rotated triples.',
             'Before each repeat: 100 warmups/control + timer priming; immediately before each sample: 10 warmups.',
             'No samples excluded. Aggregate=median of all 54 samples/control; repeat ratios use repeat medians.']
    completed = 0
    for kv in (4096, 8192):
        path = directory / f'kv{kv}.json'
        if not path.exists():
            lines.append(f'KV{kv}: missing')
            continue
        try:
            report = json.loads(path.read_text())
            repeats = report.get('measurement_rounds', [])
            settings = report.get('settings', {})
            expected = dict(rounds=1, timing_rounds=6, pairs=9, initial_warmup=100, warmup=10)
            complete = (report['status'] == 'passed' and report.get('correctness') == 'passed'
                        and all(settings.get(k) == v for k, v in expected.items()) and len(repeats) == 6
                        and report.get('timer_primed') is True
                        and len(report.get('summary', [])) == 1
                        and report['summary'][0]['static_builds'] == 1
                        and {r['measurement_round'] for r in repeats} == set(range(1, 7))
                        and all(r['compile_round'] == 1 and set(r['samples_us']) == set(LABELS)
                                and all(len(r['samples_us'][label]) == 9 for label in LABELS) for r in repeats))
            lines.append(f"KV{kv}: status={report['status']} phase={report.get('phase')} repeats={len(repeats)}/6 complete={complete}")
            if complete:
                all_samples = {label: [v for r in repeats for v in r['samples_us'][label]] for label in LABELS}
                med = {label: statistics.median(values) for label, values in all_samples.items()}
                lines.append('AGGREGATE_us ' + json.dumps(med, sort_keys=True))
                for control in ('fixed_r8', 'ascendc'):
                    ratios = [r['median_us'][control] / r['median_us']['static'] for r in repeats]
                    lines.append(f"{control}/prefetch overall={med[control]/med['static']:.6f} "
                                 f"repeat_median={statistics.median(ratios):.6f} "
                                 f"range=[{min(ratios):.6f},{max(ratios):.6f}] wins={sum(x>1 for x in ratios)}/6")
                completed += 1
            for r in repeats:
                med = r['median_us']
                lines.append(f"REPEAT {r['measurement_round']} prefetch={med['static']:.3f} fixed_R8={med['fixed_r8']:.3f} "
                             f"AscendC={med['ascendc']:.3f} fixed/prefetch={med['fixed_r8']/med['static']:.6f} "
                             f"AC/prefetch={med['ascendc']/med['static']:.6f}")
                lines.append('SAMPLES_us ' + json.dumps({label: [round(v, 3) for v in values]
                                                        for label, values in r['samples_us'].items()}, sort_keys=True))
        except (ValueError, KeyError, TypeError) as exc:
            lines.append(f'KV{kv}: invalid report {exc}')
    lines.insert(0, f"status={'passed' if completed == 2 else 'incomplete_or_error'} completed={completed}/2")
    output = directory.parent / 'mha_merge_prefetch_confirm_paste.txt'
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
